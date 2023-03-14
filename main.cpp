#include "ggml.h"
#include "llama.h"
#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

static const int EOS_TOKEN_ID = 2;

// determine number of model parts based on the dimension
static const std::map<int, int> LLAMA_N_PARTS = {
    { 4096, 1 },
    { 5120, 2 },
    { 6656, 4 },
    { 8192, 8 },
};


static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    printf(ANSI_COLOR_RESET);
    printf("\n"); // this also force flush stdout.
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif


int main(int argc, char ** argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/7B/ggml-model-q4_0.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

//    params.prompt = R"(// this function checks if the number n is prime
//bool is_prime(int n) {)";

    int64_t t_load_us = 0;

    // load the model
    llama_context* ctx_ptr = llama_init_from_params(params);
    llama_context & ctx = *ctx_ptr;
    gpt_vocab & vocab = llama_context_get_vocab(ctx);

    // print system information
    llama_print_context_info(ctx);

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');
    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = llama_tokenize_text(ctx, params.prompt);

    // prefix & suffix for instruct mode
    const std::vector<gpt_vocab::id> inp_pfx = ::llama_tokenize(vocab, "\n\n### Instruction:\n\n", true);
    const std::vector<gpt_vocab::id> inp_sfx = ::llama_tokenize(vocab, "\n\n### Response:\n\n", false);

    // in instruct mode, we inject a prefix and a suffix to each input by the user
    if (params.instruct) {
        params.interactive = true;
        params.antiprompt.push_back("### Instruction:\n\n");
    }

    // tokenize the reverse prompt
    std::vector<gpt_vocab::id> antiprompt_inp = llama_tokenize_text(ctx, params.prompt);

    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        signal(SIGINT, sigint_handler);
#endif

        fprintf(stderr, "%s: interactive mode on.\n", __func__);

        if(antipromptv_inp.size()) {
            for (size_t apindex = 0; apindex < antipromptv_inp.size(); ++apindex) {
                auto antiprompt_inp = antipromptv_inp.at(apindex);
                fprintf(stderr, "%s: reverse prompt: '%s'\n", __func__, params.antiprompt.at(apindex).c_str());
                fprintf(stderr, "%s: number of tokens in reverse prompt = %zu\n", __func__, antiprompt_inp.size());
                for (int i = 0; i < (int) antiprompt_inp.size(); i++) {
                    fprintf(stderr, "%6d -> '%s'\n", antiprompt_inp[i], vocab.id_to_token.at(antiprompt_inp[i]).c_str());
                }
                fprintf(stderr, "\n");
            }
        }
    }
    fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "\n\n");

    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n\n");
        is_interacting = true;
    }

    bool input_noecho = false;

    int remaining_tokens = params.n_predict;

    // set the color for the prompt which will be output initially
    if (params.use_color) {
        printf(ANSI_COLOR_YELLOW);
    }

    if(!llama_injest_input(ctx, params.prompt))
    {
        fprintf(stderr, "Failed to injest prompt\n");
        return 1;
    };

    // display text
    input_noecho = false;
    const std::vector<gpt_vocab::id>& embd = llama_context_get_embd(ctx);
    if (!input_noecho) {
            for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }   
        fflush(stdout);
    }

    if (!input_noecho && params.use_color) {
        printf(ANSI_COLOR_RESET);
    }

    const std::vector<gpt_vocab::id>& last_n_tokens = llama_context_get_last_n_tokens(ctx);

    while (llama_context_not_finished(ctx) > 0) {        
        gpt_vocab::id model_output = 0;
        bool response = llama_inference(ctx, model_output);
        if (response) {
            printf("%s", vocab.id_to_token[model_output].c_str());
            fflush(stdout);
        }
        // reset color to default if we there is no pending user input
        if (!input_noecho && params.use_color && (int)embd_inp.size() == input_consumed) {
            printf(ANSI_COLOR_RESET);
        }


        // in interactive mode, and not currently processing queued inputs;
        // check if we should prompt the user for more
        if (params.interactive) {
            // check for reverse prompt
            for (auto antiprompt_inp : antipromptv_inp) {
                if (antiprompt_inp.size() && std::equal(antiprompt_inp.rbegin(), antiprompt_inp.rend(), last_n_tokens.rbegin())) {
                    // reverse prompt found
                    is_interacting = true;
                    break;
                }
            }
            if (is_interacting) {
                if (params.instruct) {
                    input_consumed = embd_inp.size();
                    embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());

                    printf("\n> ");
                }

                // currently being interactive
                if (params.use_color) printf(ANSI_BOLD ANSI_COLOR_GREEN);
                std::string buffer;
                std::string line;
                bool another_line = true;
                do {
                    std::getline(std::cin, line);
                    if (line.empty() || line.back() != '\\') {
                        another_line = false;
                    } else {
                        line.pop_back(); // Remove the continue character
                    }
                    // Do not clear existing context in interactive mode
                    llama_init_context_with_prompt(ctx, buf, false);
                }

                remaining_tokens -= line_inp.size();

                input_noecho = true; // do not echo this again
            }
            is_interacting = false;
        }

        // end of text token
        if (embd.back() == EOS_TOKEN_ID) {
            if (params.interactive) {
                is_interacting = true;
            } else {
                fprintf(stderr, " [end of text]\n");
                break;
            }
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        if (params.interactive && remaining_tokens <= 0) {
            remaining_tokens = params.n_predict;
            is_interacting = true;
        }
    }
    
    // report timing from context
    {
        const int64_t t_main_end_us = ggml_time_us();
        llama_print_end_stats(ctx);
        fprintf(stderr, "%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }
    llama_free_context(ctx_ptr);    

    if (params.use_color) {
        printf(ANSI_COLOR_RESET);
    }

    return 0;
}
