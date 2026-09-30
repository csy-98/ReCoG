# 🧭 ReCoG methodology

ReCoG combines a graph of pair-specific discriminative descriptions with
retrieval and frozen vision-language reasoning. The main setting uses one
support image for each personalized concept.

## 🌱 1. Build the contrast graph

Let concept $c_i$ have support image $I_i$ and CLIP image embedding $z_i$.
The graph contains one node per concept. Its directed edge

$$e_{i\to j}=\text{features of }c_i\text{ that distinguish it from }c_j$$

is a list of textual cues. In general, $e_{i\to j}\ne e_{j\to i}$.
The two directions must retain their concept identities, even when a lookup
requests the pair in reverse order.

For every unordered pair $(c_i,c_j)$, a frozen VLM receives the two support
images and extracts three fine-grained physical cues for each concept relative
to the other. The prompt asks it to exclude generic shared properties and
background context. One call yields both directions.

The complete graph therefore requires $N(N-1)/2$ pair extraction calls and stores
$N(N-1)$ directed cue lists. This is an offline step: a query reads existing
evidence rather than reconstructing the full graph.

In code, `ContrastGraph.pairs` groups the two directions in a `PairEvidence`
record under a sorted pair key. This storage key does **not** make the evidence
undirected. `oriented(a, b)` returns the correct A-relative-to-B and B-relative-to-A
lists. Optional shared cues can be attached to a pair and are explicitly marked
as non-discriminative in the comparison prompt.

**Read:** [`schemas.py`](../recog/schemas.py), [`graph.py`](../recog/graph.py),
[`extraction_prompt`](../recog/prompts.py).

## 🔎 2. Retrieve the Top-3 candidates

Encode the query and all support images with the same frozen CLIP image encoder.
The source configuration uses `openai/clip-vit-large-patch14-336`.
Normalize the vectors and compute:

$$s(q,c_i)=\frac{z_q^\top z_i}{\lVert z_q\rVert_2\lVert z_i\rVert_2}.$$

Sort by descending similarity and retain the first three concepts:

$$C_q=\operatorname{Top3}_{c_i\in V}\;s(q,c_i).$$

With normalized embeddings, cosine similarity equals the inner product used by
a flat FAISS index. The compact implementation expresses that calculation
directly, without shipping an index. Retrieval accepts the query embedding and
support concepts; it has no question-text or target-label input.

The one-support-per-concept contract makes support ranking and concept ranking
identical. Fewer than three available concepts yield a shorter shortlist. Exactly
equal similarities preserve the supplied concept order. Empty or zero-norm
embeddings, nonfinite values, duplicate concept identifiers, and mismatched
dimensions are rejected.

**Read:** [`retrieve_top3`](../recog/retrieval.py).

## ⚖️ 3. Match every candidate pair

For every unordered candidate pair $(c_i,c_j)$, the frozen judge receives:

1. The query image.
2. The support image of $c_i$.
3. The support image of $c_j$.
4. Both directional descriptions $e_{i\to j}$ and $e_{j\to i}$.

The prompt asks which support concept matches the query, grounding the decision
in the distinctive cues while ignoring incidental pose, background, and lighting.
The included MiniCPM adapter parses the JSON `Answer` field into exactly one of the two concept
identifiers. A malformed decision raises an error rather than silently awarding
a win. All required graph evidence is checked before judge calls begin.

With three candidates, there are $\binom{3}{2}=3$ comparisons. Pair-specific
evidence matters: the features of A used against B can differ from those used
against C.

**Read:** [`comparison_prompt`](../recog/prompts.py),
[`PairJudge`](../recog/matching.py).

## 🏅 4. Aggregate the league

Each comparison awards one win to its selected concept:

$$W(c_i)=\sum_{c_j\in C_q\setminus\{c_i\}}
\mathbf{1}[\operatorname{judge}(q,c_i,c_j)=c_i].$$

Select the concept with the largest win count. Among tied concepts, select the
largest retrieval similarity. If both quantities are exactly equal, preserve
candidate order. A three-way cycle is therefore resolved by the retrieval score.

The prediction contains the winning concept, ordered candidates and their
similarities, each match winner, and the complete win table. One candidate wins
without a comparison; an empty shortlist is rejected. A winner is relative to
the supplied concepts: the method here has no unknown-class rejection threshold.

**Read:** [`run_league`](../recog/matching.py), [`recognize`](../recog/pipeline.py).

## 🔌 Adapter contracts

| Boundary | Caller provides | Expected output |
| :--- | :--- | :--- |
| Visual encoder | Query/support images, identical CLIP preprocessing and checkpoint | Same-dimensional nonzero image embeddings |
| `EdgeExtractor` | Support images in A/B order and the extraction prompt | `PairEvidence` with the two identities and three cues per direction |
| `PairJudge` | Query/A/B images in that order and the comparison prompt | One exact concept identifier parsed from JSON `Answer` |

`support_image` and `query_image` are opaque string handles. The adapters decide
how those handles resolve to actual image content. The runnable integration in [`runtime.py`](../recog/runtime.py) resolves local
images and parses responses, while [`r2p/model_interface.py`](../r2p/model_interface.py)
loads the frozen MiniCPM model. [`cli.py`](../recog/cli.py) connects these components
to local database construction and recognition. Test doubles are confined to tests;
the CLI uses actual CLIP and MiniCPM models. See [Running](running.md).

## 🧮 Cost and limits

| Stage | Work |
| :--- | :--- |
| Offline complete graph | $\binom{N}{2}$ VLM extraction calls |
| Online retrieval | $O(Nd)$ scoring and $O(N\log N)$ sorting in this compact implementation |
| Online reasoning | $\binom{k}{2}$ judge calls, with $k\leq3$ |

A retrieval miss cannot be recovered by the league: it only compares shortlisted
concepts. Incorrect or weak edge descriptions can also mislead the judge. The
complete graph requires quadratic offline construction; sparse-graph policies
are outside the core presented here.
