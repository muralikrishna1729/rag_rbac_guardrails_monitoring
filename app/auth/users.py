

def get_user_role(username:str)->str:
    USERS = {
        "alice": "finance",
        "bob": "hr",
        "charlie": "marketing",
        "david": "engineering",
        "eve": "general",
        "sagar": "admin",  # Admin usually has access to all, or 'general'
    }
    user_key = username.lower()
    return USERS.get(user_key,None)

if __name__ == "__main__":
    test_users = ["alice", "sagar", "unknown_user"]
    
    for user in test_users:
        role = get_user_role(user)
        print(f"User: {user} -> Role: {role}")
