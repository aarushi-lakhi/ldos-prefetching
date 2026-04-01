from LLMWrapper import get_wrapper

w = get_wrapper("claude-haiku4.5")
response = w.send_pdf("What is your name????", None)

print(response)