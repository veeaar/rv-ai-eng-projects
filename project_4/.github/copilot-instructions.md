# AI Agent Instructions for Deep Research Project

## Project Overview
This is a Jupyter notebook-based project focused on implementing and exploring various reasoning techniques with large language models. The project demonstrates inference-time scaling methods, training approaches for reasoning models, and building a deep research agent.

## Key Components

### Environment Setup
- Project uses conda environment defined in `environment.yaml`
- Primary Python version: 3.11
- Key dependencies:
  - LLM Integration: `ollama-python`, `openai`, `langchain`
  - ML Framework: `torch`, `transformers`
  - Web Search: `ddgs`
  - Jupyter ecosystem: `jupyter`, `ipykernel`

### Development Workflow
1. Environment Setup:
```bash
conda env create -f environment.yaml
conda activate deep_research
python -m ipykernel install --user --name=deep_research --display-name "deep_research"
```

2. LLM Models Setup:
- Project uses Ollama models: `llama3.2:3b` and `deepseek-r1:8b`
- Optional models: `qwen2.5:3b-instruct`, `phi4-mini`

### Project Structure
Main components in `deep_research.ipynb`:
1. Environment setup (Section 1)
2. Inference-time scaling implementations (Section 2)
   - Few-shot & zero-shot Chain-of-Thought (CoT)
   - Self-consistency
   - Sequential revisions
   - Tree-of-Thought
3. STaR training approach (Section 3)
4. Deep research agent implementation (Section 4)
5. Optional multi-agent extensions (Section 5)

## Development Patterns

### Notebook Cell Flow
- Code cells are designed to be run sequentially
- Each major section builds on concepts from previous sections
- All cell outputs should be cleared before committing changes

### Code Conventions
1. Chain-of-Thought Implementations:
```python
def cot_answer(question, temperature=1.0):
    # Always include step-by-step reasoning in prompts
    # Return both reasoning chain and final answer
```

2. Agent Tools:
```python
from langchain.tools import Tool
# Always provide clear description for tools
# Include input/output specifications in docstrings
```

3. Model Interactions:
```python
MODEL = "llama3.2:3b"  # Define model at top of cell
client = OpenAI(api_key = "ollama", base_url = "http://localhost:11434/v1")
```

### Debugging
- Use notebook cell outputs to verify each step
- For agent debugging, examine the intermediate reasoning steps and tool calls
- Monitor Ollama server logs when model responses seem incorrect

## Integration Points
1. Web Search Integration:
   - Uses DuckDuckGo search via `ddgs` package
   - Search results are processed and returned as concatenated snippets

2. LLM Integration:
   - Primary: Ollama local models via OpenAI-compatible API
   - Alternative: Can use actual OpenAI API by changing client configuration

3. Langchain Integration:
   - Uses `AgentType.OPENAI_FUNCTIONS` for tool integration
   - Follows ReAct pattern for agent reasoning and tool use

## Common Workflows
1. Adding New Reasoning Methods:
   - Implement core logic in a dedicated function
   - Add demonstration cells with example usage
   - Include evaluation/comparison with existing methods

2. Extending Agent Capabilities:
   - Define new tools using Langchain Tool class
   - Update agent initialization with new tools
   - Test with representative queries

## Known Patterns
1. Model Usage:
   - Use temperature=0 for deterministic tasks
   - Use temperature>0 with multiple samples for creative tasks
   - Always provide clear reasoning instructions in prompts

2. Error Handling:
   - Verify Ollama server is running before model calls
   - Handle web search failures gracefully
   - Validate tool inputs/outputs

Remember to always test new implementations with both successful and edge cases, and maintain the educational focus of the notebook format.