# Commercial AI Agent
This is a very helpful agent that simulates a commercial agent that help you to respond any questions about kavak

## Architecture

To document the project it was used C4 model for the first to levels, C1 and C2 wil be shown as the ohgh level overview, later on Agend node and tools, will be shared as the similar to the C3 level

### High Level Diagarams
Here it is detailed how the user will interact with the system and the external LLM serice that will provide from Google.

![C1 Level](/assets/C1-level.png)

Here in the C2 level, it shows that system uses a single backend service and 2 databases, one to save all the chat history and checkpoints, the second one to store all the car details, the same data as provided in the csv file.

![C2 Level](/assets/C2-level.png)

### Agent nodes diagram

The next figure shows how it divides the solution into one single agent orchestrator and 3 different to achive the three main tasks; tell about offer value from the company, show results from different car based on user needs and get results about financial plans regarding the car the client wants to acquire.

On the left side we can see all the nodes to process a message from user to identify if it is an attack, to summarize the conversation and to identify the intention of the user.

Later on the right side it is described how the the rest of the nodes are connected to achive the three main goals.

![C3 Level](/assets/C3-level.png)

### Prompt guidance

To achive the best results it ws used a mixed of different prompt engineering strategies:
* Rol definition: For almost all prompts
* Tone control: To create kindly responses to users
* Chain of though: To  evaluate if the information is in the data provided and gives a map on how to operate internally
* Zero shot, one shot and few shots.: To improve the output of the half of prompts
* Input data definition: To help the LLM understand the  input data and how to manage the results
* Constraints and Guardrails: To keep the output focused on the expected result and avoid unexpected situations as responding to something is not desired or not part of the output
* Formatting: To provide a format that later will be used in the whatsapp messages
* Conditional Logic and Prority Logic: to help the LLM understand how to behave in some escenarios and which scenarios are more relevant
* Output Surpression: To remove undesired outputs

## Future

### For production
Due to time and scope of the solution it was missing some points that can help to deploy this solution to a production environment. To achive this, it was designed a roadmpa that will help this solution have better performance, tracebility and confidence on the process.

In the figure below it shows how the roadmap will achive some goals that helps the system has automated tools to CI and CD processes, also improves responses on the agent, and reduce security risks.
Also it includes reduce hallucination by adding RAG strategy with a Vector DB and adding human-in-the-loop step on the financial plans previous to send to the user.

![Roadmap](/assets/Agent-Roadmap.png)

### For Perfomance and quaility assurance
In the project we gurantee the performance by doing unit test on the project, but we need to reach out some other parts of the performance and quaility. We need to include integration testing to do this we can integrate some manual flow to create an automatio, based on some steps of the roadmpa, that help us to create an environment, run a automate process using and LLM that will interact with our system and store all the inputs and outputs of the system, later it will evaluate if all the responses are related to the ones that we expect, and finally will create metrics.Also we can use DeepEval or some other tools that can help us to have metrics about our system.

If we implement this, we can ensure that a new version of the agent will not have any issue or reduce the functionality of it and also will help us to mantaina code and LLM quaility.

## How to reproduce it in local

1. Build the image

```bash
docker build -t bot-image
```

2. Set the env vars you received in the email

3. Run this code

```bash
docker run -it  bot-image --port 8080:8080 --env PORT=8080 \
    --env HOST=0.0.0.0 \
    --env GOOGLE_API_KEY=$GOOGLE_API_KEY \
    --env LOG_LEVEL=$LOG_LEVEL \
    --env TWILIO_ACCOUNT_SID=$TWILIO_ACCOUNT_SID \
    --env TWILIO_AUTH_TOKEN=$TWILIO_AUTH_TOKEN \
    --env CATALOG_DB_HOST=$CATALOG_DB_HOST \
    --env CATALOG_DB_PORT=$CATALOG_DB_PORT \
    --env CATALOG_DB_USER=$CATALOG_DB_USER \
    --env CATALOG_DB_PASSWORD=$CATALOG_DB_PASSWORD \
    --env CATALOG_DB_NAME=$CATALOG_DB_NAME \
    --env DB_HOST=$DB_HOST \
    --env DB_PORT=$DB_PORT \
    --env DB_USER=$DB_USER \
    --env DB_PASSWORD=$DB_PASSWORD \
    --env DB_NAME=$DB_NAME \

```

4. Enjoy it!

### Use from cloud

You can ue the system with this endpoint

https://comm-ai-agent-279566078903.northamerica-south1.run.app/api/chat

And see the doc here:

https://comm-ai-agent-279566078903.northamerica-south1.run.app/docs
