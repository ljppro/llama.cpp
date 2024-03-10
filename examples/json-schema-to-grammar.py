#!/usr/bin/env python3
import argparse
import itertools
import json
import re
import sys
from typing import Any, Dict, List, Set, Tuple

# whitespace is constrained to a single space char to prevent model "running away" in
# whitespace. Also maybe improves generation quality?
SPACE_RULE = '" "?'

PRIMITIVE_RULES = {
    'boolean': '("true" | "false") space',
    'number': '("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? space',
    'integer': '("-"? ([0-9] | [1-9] [0-9]*)) space',
    'value'  : 'object | array | string | number | boolean',
    'object' : '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
    'array'  : '"[" space ( value ("," space value)* )? "]" space',
    'string': r''' "\"" (
        [^"\\] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
      )* "\"" space ''',
    'null': '"null" space',
}

INVALID_RULE_CHARS_RE = re.compile(r'[^a-zA-Z0-9-]+')
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n]')
GRAMMAR_LITERAL_ESCAPES = {'\r': '\\r', '\n': '\\n'}


class SchemaConverter:
    def __init__(self, prop_order):
        self._prop_order = prop_order
        self._rules = {'space': SPACE_RULE}
        self._refs = {}
        self._refs_being_resolved = set()

    def _format_literal(self, literal):
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0)), json.dumps(literal)[1:-1]
        )
        return f'"{escaped}"'

    def _add_rule(self, name, rule):
        esc_name = INVALID_RULE_CHARS_RE.sub('-', name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while f'{esc_name}{i}' in self._rules and self._rules[f'{esc_name}{i}'] != rule:
                i += 1
            key = f'{esc_name}{i}'
        self._rules[key] = rule
        return key

    def resolve_refs(self, schema: dict, url: str):
        '''
            Resolves all $ref fields in the given schema, fetching any remote schemas,
            replacing $ref with absolute reference URL and populating self._refs with the
            respective referenced (sub)schema dictionaries.
        '''
        def visit(n: dict):
            if isinstance(n, list):
                return [visit(x) for x in n]
            elif isinstance(n, dict):
                ref = n.get('$ref')
                if ref is not None and ref not in self._refs:
                    if ref.startswith('https://'):
                        import requests
                        
                        frag_split = ref.split('#')
                        base_url = frag_split[0]

                        target = self._refs.get(base_url)
                        if target is None:
                            target = self.resolve_refs(requests.get(ref).json(), base_url)
                            self._refs[base_url] = target

                        if len(frag_split) == 1 or frag_split[-1] == '':
                            return
                    elif ref.startswith('#/'):
                        target = schema
                        ref = f'{url}{ref}'
                        n['$ref'] = ref
                    else:
                        raise ValueError(f'Unsupported ref {ref}')
                    
                    for sel in ref.split('#')[-1].split('/')[1:]:
                        assert target is not None and sel in target, f'Error resolving ref {ref}: {sel} not in {target}'
                        target = target[sel]
                    
                    self._refs[ref] = target
                else:
                    for v in n.values():
                        visit(v)
            
            return n
        return visit(schema)

    def _generate_union_rule(self, name, alt_schemas):
        return ' | '.join((
            self.visit(alt_schema, f'{name}{"-" if name else ""}{i}')
            for i, alt_schema in enumerate(alt_schemas)
        ))

    def _format_range_char(self, c):
        if c in ('-', ']', '\\'):
            return '\\' + chr(c)
        elif c == '\n':
            return '\\n'
        elif c == '\r':
            return '\\r'
        elif c == '\t':
            return '\\t'
        else:
            return c

    def _visit_pattern(self, pattern, name):
        assert pattern.startswith('^') and pattern.endswith('$'), 'Pattern must start with "^" and end with "$"'
        pattern = pattern[1:-1]
        sub_rule_ids = {}
        try:
            def visit_seq(seq):
                out = []
                for t, g in itertools.groupby(seq, lambda x: x[0]):
                    g = list(g)
                    # Merge consecutive literals
                    if t == re._parser.LITERAL and len(g) > 1:
                        out.append(self._format_literal(''.join(chr(x[1]) for x in g)))
                    else:
                        out.extend(visit(x) for x in g)
                if len(out) == 1:
                    return out[0]
                return '(' + ' '.join(out) + ')'
            
            def visit(pattern):
                nonlocal sub_rule_ids

                if pattern[0] == re._parser.LITERAL:
                    return json.dumps(chr(pattern[1]))
                
                elif pattern[0] == re._parser.NOT_LITERAL:
                    return f'[^{self._format_range_char(chr(pattern[1]))}]'
                
                elif pattern[0] == re._parser.ANY:
                    raise ValueError('Unsupported pattern: "."')
                
                elif pattern[0] == re._parser.IN:
                    def format_range_comp(c):
                        if c[0] == re._parser.LITERAL:
                            return self._format_range_char(chr(c[1]))
                        elif c[0] == re._parser.RANGE:
                            return f'{self._format_range_char(chr(c[1][0]))}-{self._format_range_char(chr(c[1][1]))}'
                        else:
                            raise ValueError(f'Unrecognized pattern: {c}')
                    return f'[{"".join(format_range_comp(c) for c in pattern[1])}]'
                
                elif pattern[0] == re._parser.BRANCH:
                    return '(' + ' | '.join((visit(p) for p in pattern[1][1])) + ')'
                
                elif pattern[0] == re._parser.SUBPATTERN:
                    return '(' + visit(pattern[1][3]) + ')'
                
                elif pattern[0] == re._parser.MAX_REPEAT:
                    min_times = pattern[1][0]
                    max_times = pattern[1][1] if not pattern[1][1] == re._parser.MAXREPEAT else None
                    sub = visit(pattern[1][2])

                    id = sub_rule_ids.get(sub)
                    if id is None:
                        id = self._add_rule(f'{name}-{len(sub_rule_ids) + 1}', sub)
                        sub_rule_ids[sub] = id
                    sub = id

                    if min_times == 0 and max_times is None:
                        return f'{sub}*'
                    elif min_times == 0 and max_times == 1:
                        return f'{sub}?'
                    elif min_times == 1 and max_times is None:
                        return f'{sub}+'
                    else:
                        return ' '.join([sub] * min_times + 
                                        ([f'{sub}?'] * (max_times - min_times) if max_times is not None else [f'{sub}*']))
                
                elif isinstance(pattern, re._parser.SubPattern):
                    return visit_seq(pattern.data)
                
                elif isinstance(pattern, list):
                    return visit_seq(pattern)
                
                else:
                    raise ValueError(f'Unrecognized pattern: {pattern} ({type(pattern)})')

            return self._add_rule(name, visit(re._parser.parse(pattern)))
        except BaseException as e:
            raise Exception(f'Error processing pattern: {pattern}: {e}') from e

    def _resolve_ref(self, ref):
        ref_name = ref.split('/')[-1]
        if ref_name not in self._rules and ref not in self._refs_being_resolved:
            self._refs_being_resolved.add(ref)
            resolved = self._refs[ref]
            ref_name = self.visit(resolved, ref_name)
            self._refs_being_resolved.remove(ref)
        return ref_name
    
    def visit(self, schema, name):
        schema_type = schema.get('type')
        rule_name = name or 'root'

        if (ref := schema.get('$ref')) is not None:
            return self._resolve_ref(ref)

        elif 'oneOf' in schema or 'anyOf' in schema:
            return self._add_rule(rule_name, self._generate_union_rule(name, schema.get('oneOf') or schema['anyOf']))
        
        elif isinstance(schema_type, list):
            return self._add_rule(rule_name, self._generate_union_rule(name, [{'type': t} for t in schema_type]))

        elif 'const' in schema:
            return self._add_rule(rule_name, self._format_literal(schema['const']))

        elif 'enum' in schema:
            rule = ' | '.join((self._format_literal(v) for v in schema['enum']))
            return self._add_rule(rule_name, rule)

        elif schema_type in (None, 'object') and 'properties' in schema:
            required = set(schema.get('required', []))
            properties = list(schema['properties'].items())
            return self._add_rule(rule_name, self._build_object_rule(properties, required, name))

        elif schema_type in (None, 'object') and 'allOf' in schema:
            required = set()
            properties = []
            hybrid_name = name
            def add_component(comp_schema, is_required):
                if (ref := comp_schema.get('$ref')) is not None:
                    comp_schema = self._refs[ref]
                
                if 'properties' in comp_schema:
                    for prop_name, prop_schema in comp_schema['properties'].items():
                        properties.append((prop_name, prop_schema))
                        if is_required:
                            required.add(prop_name)

            for t in schema['allOf']:
                if 'anyOf' in t:
                    for tt in t['anyOf']:
                        add_component(tt, is_required=False)
                else:
                    add_component(t, is_required=True)

            return self._add_rule(rule_name, self._build_object_rule(properties, required, hybrid_name))

        elif schema_type in (None, 'object') and 'additionalProperties' in schema:
            additional_properties = schema['additionalProperties']
            if not isinstance(additional_properties, dict):
                additional_properties = {}

            sub_name = f'{name}{"-" if name else ""}additionalProperties'
            value_rule = self.visit(additional_properties, f'{sub_name}-value')
            kv_rule = self._add_rule(f'{sub_name}-kv', f'string ":" space {value_rule}')
            return self._add_rule(
                rule_name,
                f'( {kv_rule} ( "," space {kv_rule} )* )*')

        elif schema_type in (None, 'array') and 'items' in schema:
            # TODO `prefixItems` keyword
            items = schema['items']
            if isinstance(items, list):
                return self._add_rule(
                    rule_name,
                    '"[" space ' +
                    ' "," space '.join(
                        self.visit(item, f'{name}-{i}')
                        for i, item in enumerate(items)) +
                    ' "]" space')
            else:
                item_rule_name = self.visit(items, f'{name}{"-" if name else ""}item')
                list_item_operator = f'( "," space {item_rule_name} )'
                successive_items = ""
                min_items = schema.get("minItems", 0)
                max_items = schema.get("maxItems")
                if min_items > 0:
                    successive_items = list_item_operator * (min_items - 1)
                    min_items -= 1
                if max_items is not None and max_items > min_items:
                    successive_items += (list_item_operator + "?") * (max_items - min_items - 1)
                else:
                    successive_items += list_item_operator + "*"
                if min_items == 0:
                    rule = f'"[" space ( {item_rule_name} {successive_items} )? "]" space'
                else:
                    rule = f'"[" space {item_rule_name} {successive_items} "]" space'
                return self._add_rule(rule_name, rule)
            
        elif schema_type in (None, 'string') and 'pattern' in schema:
            return self._visit_pattern(schema['pattern'], rule_name)

        elif schema_type in (None, 'string') and re.match(r'^uuid[1-5]?$', schema.get('format', '')):
            return self._visit_pattern('^([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$', 'uuid')

        elif schema_type == 'object' and len(schema) == 1 or schema_type is None and len(schema) == 0:
            # This depends on all primitive types
            for t, r in PRIMITIVE_RULES.items():
                self._add_rule(t, r)
            return 'object'
        
        else:
            assert schema_type in PRIMITIVE_RULES, f'Unrecognized schema: {schema}'
            return self._add_rule(
                'root' if rule_name == 'root' else schema_type,
                PRIMITIVE_RULES[schema_type]
            )
    
    def _build_object_rule(self, properties: List[Tuple[str, Any]], required: Set[str], name: str):
        prop_order = self._prop_order
        # sort by position in prop_order (if specified) then by original order
        sorted_props = [kv[0] for _, kv in sorted(enumerate(properties), key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]))]

        prop_kv_rule_names = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(prop_schema, f'{name}{"-" if name else ""}{prop_name}')
            prop_kv_rule_names[prop_name] = self._add_rule(
                f'{name}{"-" if name else ""}{prop_name}-kv',
                fr'{self._format_literal(prop_name)} space ":" space {prop_rule_name}'
            )

        required_props = [k for k in sorted_props if k in required]
        optional_props = [k for k in sorted_props if k not in required]
        
        rule = '"{" space '
        rule += ' "," space '.join(prop_kv_rule_names[k] for k in required_props)            

        if optional_props:
            rule += ' ('
            if required_props:
                rule += ' "," space ( '

            def get_recursive_refs(ks, first_is_optional):
                [k, *rest] = ks
                kv_rule_name = prop_kv_rule_names[k]
                if first_is_optional:
                    res = f'( "," space {kv_rule_name} )?'
                else:
                    res = kv_rule_name
                if len(rest) > 0:
                    res += ' ' + self._add_rule(
                        f'{name}{"-" if name else ""}{k}-rest',
                        get_recursive_refs(rest, first_is_optional=True)
                    )
                return res

            rule += ' | '.join(
                get_recursive_refs(optional_props[i:], first_is_optional=False)
                for i in range(len(optional_props))
            ) + ' '
            if required_props:
                rule += ' ) '
            rule += ' )? '

        rule += ' "}" space '

        return rule

    def format_grammar(self):
        return '\n'.join((f'{name} ::= {rule}' for name, rule in self._rules.items()))


def main(args_in = None):
    parser = argparse.ArgumentParser(
        description='''
            Generates a grammar (suitable for use in ./main) that produces JSON conforming to a
            given JSON schema. Only a subset of JSON schema features are supported; more may be
            added in the future.
        ''',
    )
    parser.add_argument(
        '--prop-order',
        default=[],
        type=lambda s: s.split(','),
        help='''
            comma-separated property names defining the order of precedence for object properties;
            properties not specified here are given lower precedence than those that are, and
            are kept in their original order from the schema. Required properties are always
            given precedence over optional properties.
        '''
    )
    parser.add_argument('schema', help='file containing JSON schema ("-" for stdin)')
    args = parser.parse_args(args_in)

    if args.schema.startswith('https://'):
        url = args.schema
        import requests
        schema = requests.get(url).json()
    elif args.schema == '-':
        url = 'stdin'
        schema = json.load(sys.stdin)
    else:
        url = f'file://{args.schema}'
        with open(args.schema) as f:
            schema = json.load(f)
    prop_order = {name: idx for idx, name in enumerate(args.prop_order)}
    converter = SchemaConverter(prop_order)
    schema = converter.resolve_refs(schema, url)
    converter.visit(schema, '')
    print(converter.format_grammar())


if __name__ == '__main__':
    main()
