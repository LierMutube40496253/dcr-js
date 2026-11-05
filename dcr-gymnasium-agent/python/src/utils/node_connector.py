def connect_to_dcr_environment(env_url="http://localhost:5001"):
    import requests

    # One keep-alive connection for all calls (special-course): a new TCP connection
    # per step exhausts Windows' ephemeral ports (WinError 10048) in long runs.
    session = requests.Session()

    def reset():
        r = session.post(f"{env_url}/reset")
        r.raise_for_status()
        return r.json()

    def replay(events):
        # Special course (partway starts): an illegal event comes back as HTTP 400 with its name
        r = session.post(f"{env_url}/replay", json={"events": list(events)})
        if r.status_code >= 400:
            raise RuntimeError(f"replay failed: {r.text}")
        return r.json()

    def get_state():
        r = session.get(f"{env_url}/state")
        r.raise_for_status()
        return r.json()

    def send_action(action):
        r = session.post(f"{env_url}/action", json={"action": action})
        r.raise_for_status()
        return r.json()

    return {
        "reset": reset,
        "replay": replay,
        "get_state": get_state,
        "send_action": send_action
    }