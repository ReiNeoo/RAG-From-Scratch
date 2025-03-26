# class SessionMemory:
#     def __init__(self):
#         self.session = []

#     def update_session(self, query, response):
#         self.session.append({'user': query, 'bot': response})


class SessionBasedMemory:
    def __init__(self):
        self.session = []

    def update_session(self, query, response):
        self.session.append({'user': query, 'bot': response})

    def response_listener(self, query: str, response_list: list) -> dict:
        response = self._create_full_response(response_list)
        self.update_session(query, response)

    def generate_prompt(self, question):
        conversation = "\n".join(
            [f"question of user: {item.get('user')}\nResponse of Bot: {item.get('bot')}" for item in self.session])
        query = f"""
            Based on the following context and conversation history, answer the query.
            Answer like a fine tuned finance LLM instead of a rag system. 
            Answer like you are an expert about given content. 

        Context:
        {conversation}
        
        Query: {question}

            """
        return query

    def _create_full_response(self, response_list):
        return ''.join(response_list)
