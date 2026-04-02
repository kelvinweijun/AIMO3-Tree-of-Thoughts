# 🌳 AIMO-3 Pure Tree of Thoughts Solver

A competitive mathematics problem solver built for the [AI Mathematical Olympiad (AIMO) Progress Prize 3](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3) on Kaggle. It combines a **Tree of Thoughts** (ToT) reasoning framework with a locally-served large language model (vLLM) and sandboxed Python execution to tackle IMO-level problems.

---

## Overview

Rather than generating a single chain of reasoning, this solver constructs a *tree* of candidate reasoning paths, evaluates and prunes them, then synthesizes the most promising branches into a final answer. This mirrors how expert mathematicians explore multiple approaches before committing to one.

```
Problem
  ├── Approach 1: Direct Analysis       → answer candidate
  ├── Approach 2: Pattern Recognition   → answer candidate
  └── Approach 3: Decomposition
        ├── Sub-step refinement         → answer candidate
        └── Alternative direction       → answer candidate
              └── Synthesis             → FINAL ANSWER
```

---

## Key Features

- **Pure Tree of Thoughts search** — beam search over a tree of reasoning paths with UCB-based node selection
- **Parallel exploration** — multiple reasoning branches processed concurrently via `ThreadPoolExecutor`
- **Sandboxed Python execution** — each reasoning node can call a live Jupyter kernel to run `sympy`, `numpy`, and `mpmath` for exact symbolic or numerical computation
- **Entropy-based evaluation** — logprob entropy from the LLM is used as a confidence signal for node scoring
- **Adaptive pruning** — low-quality branches are pruned each round to focus compute on promising paths
- **Frequency-first voting** — final answer is selected by a two-stage vote: answer frequency (primary) then average node score (tiebreaker)
- **Time-budget management** — dynamically allocates time per problem based on remaining problems and notebook runtime

---

## Architecture

### Components

| Component | Description |
|---|---|
| `TreeOfThoughts` | Core tree data structure; manages nodes, expansion, pruning, and beam search |
| `ThoughtNode` | Single node storing reasoning content, scores, entropy, and Python execution stats |
| `ToTPromptGenerator` | Generates diverse initial, expansion, refinement, and synthesis prompts |
| `PureToTSolver` | Orchestrates the full solve pipeline: vLLM server, Jupyter kernels, ToT search |
| `AIMO3Sandbox` | Stateful Jupyter kernel wrapper for sandboxed Python execution |
| `AIMO3Tool` | Tool adapter that feeds Python execution results back into the conversation |
| `AIMO3Template` | Formats messages using the `openai_harmony` conversation schema |
| `CFG` | Central configuration class for all hyperparameters |

### Four-Phase Search

1. **Initial Exploration** — Three diverse approaches are generated and processed in parallel
2. **Iterative Expansion** — Best nodes are expanded using UCB selection over multiple rounds; temperature decays each round
3. **Refinement** — Top leaf nodes receive focused, lower-temperature follow-up completions
4. **Synthesis** — If no clear consensus exists, a synthesis prompt combines insights from the best paths

---

## Requirements

This notebook is designed to run in the Kaggle competition environment with the following dependencies pre-loaded via a local wheel archive:

- `unsloth`
- `trl`
- `vllm`
- `openai_harmony`
- `transformers`
- `polars`
- `pandas`
- `openai`
- `jupyter_client`

Standard libraries used: `math`, `re`, `threading`, `queue`, `concurrent.futures`, `subprocess`, `gc`

The model served is `danielhanchen/gpt-oss-120b` (loaded from `/kaggle/input/`).

---

## Configuration

All hyperparameters live in the `CFG` class:

```python
# Tree of Thoughts search
CFG.tot_max_depth         = 4      # Maximum tree depth
CFG.tot_branching_factor  = 3      # Children per node
CFG.tot_beam_width        = 8      # Nodes kept after pruning
CFG.tot_pruning_threshold = 0.15   # Score fraction below which nodes are pruned
CFG.tot_expansion_rounds  = 6      # Number of iterative expansion rounds
CFG.tot_refinement_attempts = 3    # Top nodes to refine in Phase 3
CFG.tot_synthesis_enabled = True   # Enable synthesis phase

# Generation
CFG.temperature           = 0.8
CFG.min_p                 = 0.02
CFG.context_tokens        = 100000

# Compute
CFG.workers               = 16     # Parallel threads / Jupyter kernels
CFG.gpu_memory_utilization = 0.96
```

---

## Node Scoring

Each `ThoughtNode` is scored by a composite function:

```
score = (1 / entropy) × answer_bonus × error_penalty × python_bonus × quality_bonus × depth_bonus
```

- **Entropy** (from logprobs): lower = more confident model output
- **Answer bonus**: ×5 if the node produced a valid boxed answer
- **Error penalty**: ×0.5 per Python execution error
- **Quality metrics**: reasoning coherence, mathematical rigor, and progress (derived from response text heuristics)
- **UCB score** used during expansion for exploration–exploitation balance

---

## Answer Selection

Final answer selection uses **frequency-first weighted voting**:

```python
# Primary: count how many nodes agree on each answer
# Secondary: average node score as tiebreaker (prevents a single high-scoring outlier from winning)
answer_weights = {
    ans: count * avg_score
    for ans, count in answer_counts.items()
}
```

---

## Usage

The solver integrates with the Kaggle AIMO-3 inference server:

```python
solver = PureToTSolver(CFG)

def predict(id_, question, answer=None):
    final_answer = solver.solve_problem(question.item(0))
    return pl.DataFrame({'id': id_.item(0), 'answer': final_answer})

inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(('reference.csv',))
```

---

## How It Differs from Standard Chain-of-Thought

| | Chain-of-Thought | Tree of Thoughts (this solver) |
|---|---|---|
| Search strategy | Single linear path | Tree with beam search |
| Exploration | One attempt | Multiple parallel branches |
| Backtracking | None | UCB-guided node selection |
| Answer aggregation | Last answer wins | Frequency + score voting |
| Compute allocation | Fixed | Dynamic pruning + budget management |

---

## License

This project is intended for research and competition use. Model weights and competition data are subject to their respective licenses from Kaggle and the model provider.
