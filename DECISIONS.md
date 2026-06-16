# Architectural Decisions

## 1. Core Architecture: LangGraph State Machine
**Decision:** Move from a linear pipeline to a **LangGraph** directed acyclic graph (DAG).
**Rationale:** The assessment requires complex branching (False Premise -> Refusal -> Generation). A linear script becomes "spaghetti code" when handling these edge cases. LangGraph allows us to treat each step as a discrete node with its own state.

## 2. Local LLM: Qwen2.5-1.5B-Instruct
**Decision:** Run a local model instead of a cloud API.
**Rationale:** Satisfies the privacy-first requirement and avoids API quota failures. 1.5B is the "sweet spot" for 8GB RAM machines.

## 3. Three Things That Didn't Work
1.  **Zero-Shot Refusal:** Initially, I tried to let the LLM decide to refuse purely via the system prompt ("Only answer from context"). However, smaller local models (1.5B) are prone to "hallucinating" technical knowledge from their training data. I had to implement a dedicated **Context Sufficiency Grader** node to explicitly block generation if the retriever fails.
2.  **ChromaDB with Large Chunks:** Using 2000+ character chunks led to "Context Dilution" where the LLM got overwhelmed by irrelevant text, causing it to miss the specific answer. Switching to 1000-character chunks with overlap improved recall significantly.
3.  **Regular Expression JSON Parsing:** Early versions tried to parse JSON from the LLM using regex. Local models often add conversational filler ("Here is the JSON: ..."). I moved to a more robust `JsonOutputParser` and a recursive search for `{}` blocks to handle model inconsistency.

## 4. Chunking Strategy & Failure Mode
**Strategy:** `RecursiveCharacterTextSplitter` (1000 size, 100 overlap).
**Specific Failure Mode:** It still splits **large code blocks**. If a FastAPI code example is 1200 characters long, it gets cut in half. The LLM then sees incomplete syntax, leading to potentially broken code citations.

## 5. Refusal Mechanism & Weakness
**How it works:** We use a "Grader" node that looks at the retrieved documents and the query *before* the answer is generated. It produces a binary `is_sufficient` score.
**Category it gets wrong:** **"Negative overlap" questions.** If a user asks "How do I NOT use Pydantic in FastAPI?", the retriever finds many pages about "Pydantic". The grader sees the word "Pydantic" in the context and thinks it's sufficient, but the documents don't actually explain how to *avoid* it, leading to a "hallucinated" refusal or a generic answer.

## 6. With one more week and $500/month...
1.  **Reranking:** Implement a Cross-Encoder reranker (like BGE-Reranker) to filter the top 10 documents down to the best 3.
2.  **Agentic Search:** Allow the graph to loop. If the first search fails, use an LLM to "rewrite" the query and try again (Self-RAG).
3.  **Fine-tuning:** Fine-tune the 1.5B model on the FastAPI documentation specifically to improve technical accuracy.

## 7. Known Shortcut
**Shortcut:** I used a fixed cost estimate ($0.0) in the logs instead of implementing a real-time token counter (tiktoken) for the local model to stay within the 24h limit.
