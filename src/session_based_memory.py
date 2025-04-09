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
            You are a DeFi expert. Answer the following query with deep expertise, avoiding phrases like 
            'based on the given context' or 'according to the provided information'. 
            Your response should sound authoritative and natural.  

        Context:  
        {conversation}  

        Query: {question}
        """

        return query

    def _create_full_response(self, response_list):
        return ''.join(response_list)
