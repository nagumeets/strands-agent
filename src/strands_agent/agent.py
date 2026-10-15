import os
import asyncio

# Force the environment variables before importing strands
#os.environ["STRANDS_DEFAULT_MODEL"] = "bedrock/us.amazon.nova-2-lite-v1:0"

from strands import Agent
from strands.models import BedrockModel

system_prompit="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!

Question: {input}
Thought:{agent_scratchpad}"""
async def main():
    # Configure the model directly to assign max output token constraints
    nova_model = BedrockModel(
        model_id="us.amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
        max_tokens=4096,
    )

    agent = Agent(
        model=nova_model,
        system_prompt=system_prompit
    )

    print("Calling agent using Amazon Nova 2 Lite...")
    
    # FIX: Use invoke_async instead of calling the agent directly, 
    # and access the response payload via .message
    response = await agent.invoke_async("Find the capital of france and tell me the weather of that?")
    print(response.structured_output)

if __name__ == "__main__":
    asyncio.run(main())