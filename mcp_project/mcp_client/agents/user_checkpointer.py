
class UserCheckpointer:
    def __init__(self):
        self._user_states = {}

    def save(self, user_id, state):
        self._user_states[user_id] = state

    def load(self, user_id):
        return self._user_states.get(user_id)