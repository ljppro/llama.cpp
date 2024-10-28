'''
    This script fetches all the models used in the server tests.
    
    This is useful for slow tests that use larger models, to avoid them timing out on the model downloads.
    
    It is meant to be run from the root of the repository.
    
    Example:
        python scripts/fetch_server_test_models.py
        ( cd examples/server/tests && ./tests.sh --tags=slow )
'''
import os
from behave.parser import Parser
import glob
import re
from pydantic import BaseModel
import subprocess


class HuggingFaceModel(BaseModel):
    hf_repo: str
    hf_file: str

    class Config:
        frozen = True


models = set()

model_file_re = re.compile(r'a model file ([^\s\n\r]+) from HF repo ([^\s\n\r]+)')


def process_step(step):
    if (match := model_file_re.search(step.name)):
        (hf_file, hf_repo) = match.groups()
        models.add(HuggingFaceModel(hf_repo=hf_repo, hf_file=hf_file))


feature_files = glob.glob(
    os.path.join(
        os.path.dirname(__file__),
        '../examples/server/tests/features/*.feature'))

for feature_file in feature_files:
    with open(feature_file, 'r') as file:
        feature = Parser().parse(file.read())
        if not feature: continue

    if feature.background:
        for step in feature.background.steps:
            process_step(step)

    for scenario in feature.walk_scenarios(with_outlines=True):
        for step in scenario.steps:
            process_step(step)

cli_path = os.environ.get(
    'LLAMA_SERVER_BIN_PATH',
    os.path.join(
        os.path.dirname(__file__),
        '../build/bin/Release/llama-cli.exe' if os.name == 'nt' else '../build/bin/llama-cli'))

for m in models:
    if '<' in m.hf_repo or '<' in m.hf_file:
        continue
    print(f'# Ensuring model at {m.hf_repo} / {m.hf_file} is fetched')
    subprocess.check_call([cli_path, '-hfr', m.hf_repo, '-hff', m.hf_file, '-fa', '-n', '1', '-p', 'Hey', '--no-warmup'])
