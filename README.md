# LLM personal projects

## Job onboarding chatbot agent to accommodate efficient onboarding process
- This is an LLM agent example I built for job onboarding. Especially for new hires, using a chatbot agent may make their onboarding process easier and more effficient.  
- Run code lines of  `./onboarding_chatbot/Langchain_onboarding_hk.ipynb`, which will start the chatbot interactive web server.  
- `./onboarding_chatbot/app.py` includes source code behind the agent.  
- `./RAG_data` includes some example onboarding materials.   
---

## Github repo searchbot agent to find out relevant code examples
- This is an LLM agent example I built for searching relevant information from github repositories you want to search for.  

### You may take one of the two different approaches to achieve the goal  
- Approach 1: Look at `./repo_searchbot/app.py` or `./repo_searchbot/github_repo_search_4.ipynb`, both of which contain the same code.  
    - This used OpenAI SDK as an LLM agent framework.    
- Approach 2: Also, you can look at `./repo_searchbot/github_repo_search_aws_langgraph.py`.  
    - This used AWS OpenSearch Service, AWS Bedrock, and Langchain & Langgraph as agent frameworks.  

---
### References
[Langchain tutorials](https://python.langchain.com/docs/tutorials/)   
[Repo search example](https://github.com/IIEleven11/Talk2Repo)   
[Structured output in Langchain](https://python.langchain.com/docs/concepts/structured_outputs/)  
[Structured output in OpenAI](https://platform.openai.com/docs/guides/structured-outputs/introduction?api-mode=chat)  
[AWS OpenSearch Service tutorials](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/tutorials.html)  
[AWS Bedrock tutorials](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html)  

---
### Contact
Hakwoo Kim (hakwoo@gmail.com)  