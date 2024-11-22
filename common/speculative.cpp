#include "speculative.h"

#include "log.h"
#include "common.h"
#include "sampling.h"

#include <cstring>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct common_speculative {
    struct common_speculative_params params;

    llama_batch batch;

    struct llama_context * ctx;
    struct common_sampler * smpl;

    llama_tokens prompt;
};

struct common_speculative * common_speculative_init(
        struct common_speculative_params params,
        struct llama_context * ctx_dft) {
    auto * result = new common_speculative {
        /* .params = */ params,
        /* .batch  = */ llama_batch_init(llama_n_batch(ctx_dft), 0, 1),
        /* .ctx    = */ ctx_dft,
        /* .smpl   = */ nullptr,
        /* .prompt = */ {},
    };

    // TODO: optimize or pass from outside?
#if 1
    {
        common_sampler_params sparams;
        sparams.no_perf = false;

        sparams.top_k = 40;
        sparams.top_p = 0.9;

        sparams.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_INFILL,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), sparams);
    }
#else
    {
        common_sampler_params sparams;
        sparams.no_perf = false;

        sparams.top_k = 10;

        sparams.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), sparams);
    }
#endif

    return result;
}

void common_speculative_free(struct common_speculative * spec) {
    common_sampler_free(spec->smpl);

    llama_batch_free(spec->batch);

    delete spec;
}

bool common_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft) {
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);
    const struct llama_model * model_dft = llama_get_model(ctx_dft);

    const bool vocab_type_tgt = llama_vocab_type(model_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(model_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but "
                     "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_add_bos_token(model_tgt) != llama_add_bos_token(model_dft) ||
        llama_add_eos_token(model_tgt) != llama_add_eos_token(model_dft) ||
        llama_token_bos(model_tgt) != llama_token_bos(model_dft) ||
        llama_token_eos(model_tgt) != llama_token_eos(model_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_n_vocab(model_tgt);
        const int n_vocab_dft = llama_n_vocab(model_dft);

        const int vocab_diff = std::abs(n_vocab_tgt - n_vocab_dft);

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but "
                         "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    __func__, n_vocab_tgt, llama_n_vocab(model_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_token_get_text(model_tgt, i);
            const char * token_text_dft = llama_token_get_text(model_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but "
                             "token %d content differs - target '%s', draft '%s'\n", __func__, i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

void common_speculative_add_draft(
        struct common_speculative * spec,
        struct llama_batch & batch_tgt,
        const llama_tokens & prompt_tgt,
        llama_token id_last,
        llama_token n_past_tgt) {
    auto & batch  = spec->batch;
    auto & ctx    = spec->ctx;
    auto & smpl   = spec->smpl;
    auto & prompt = spec->prompt;

    int reuse_i = 0;
    int reuse_n = 0;

    const int n_ctx = llama_n_ctx(ctx) - spec->params.n_draft;

    const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

    for (int i = 0; i < (int) prompt.size(); ++i) {
        int cur = 0;
        while (i_start + cur < (int) prompt_tgt.size() &&
               i       + cur < (int) prompt.size() &&
               prompt_tgt[i_start + cur] == prompt[i + cur]) {
            cur++;
        }

        if ((cur >= spec->params.n_reuse || prompt_tgt.size() <= n_ctx) && cur > reuse_n) {
            reuse_i = i;
            reuse_n = cur;
        }
    }

    LOG_DBG("%s: reuse_i = %d, reuse_n = %d\n", __func__, reuse_i, reuse_n);

    if (reuse_n == 0) {
        llama_kv_cache_clear(ctx);

        prompt.clear();
    } else {
        llama_kv_cache_seq_rm (ctx, 0, 0, reuse_i);
        llama_kv_cache_seq_rm (ctx, 0, reuse_i + reuse_n, -1);
        llama_kv_cache_seq_add(ctx, 0, reuse_i, -1, -reuse_i);

        prompt.erase(prompt.begin(), prompt.begin() + reuse_i);
        prompt.erase(prompt.begin() + reuse_n, prompt.end());
    }

    common_batch_clear(batch);

    for (int i = i_start + reuse_n; i < (int) prompt_tgt.size(); ++i) {
        //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_tgt[i]);
        common_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false);

        prompt.push_back(prompt_tgt[i]);
    }

    const llama_pos n_past = prompt_tgt.size() - i_start;

    LOG_DBG("%s: n_past = %d\n", __func__, n_past);

    if (batch.n_tokens > 0) {
        LOG_DBG("%s: draft batch: %s\n", __func__, string_from(ctx, batch).c_str());

        llama_decode(ctx, batch);
    }

    common_batch_clear(batch);
    common_batch_add  (batch, id_last, n_past, { 0 }, true);

    prompt.push_back(id_last);

    LOG_DBG("%s: prompt_last: %s\n", __func__, string_from(ctx, prompt).c_str());

    llama_decode(ctx, batch);

    common_sampler_reset(smpl);

    // sample n_draft tokens from the draft model
    for (int i = 0; i < spec->params.n_draft; ++i) {
        common_batch_clear(batch);

        common_sampler_sample(smpl, ctx, 0, true);

        const auto * cur_p = common_sampler_get_candidates(smpl);

        for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
            LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                    k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx, cur_p->data[k].id).c_str());
        }

        // add drafted token for each sequence
        const llama_token id = cur_p->data[0].id;

        // only collect very high-confidence draft tokens
        if (cur_p->data[0].p < spec->params.p_min) {
            break;
        }

        common_sampler_accept(smpl, id, true);

        common_batch_add(batch_tgt, id, n_past_tgt + i, { 0 }, true);

        if (batch_tgt.n_tokens > spec->params.n_draft) {
            break;
        }

        common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

        // evaluate the drafted tokens on the draft model
        llama_decode(ctx, batch);

        prompt.push_back(id);
    }

    // don't waste time on small batches
    // TODO: do not evaluate the draft model for that many rounds
    if (batch_tgt.n_tokens < spec->params.n_min) {
        batch_tgt.n_tokens = 1;
    }
}
