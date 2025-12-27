from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import re

MODEL = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)


if __name__ == "__main__":
    # Prompt for the Urgency
    urgency = """Role: You are a support analyst and will classify the ticke urgency based on its content. Rules: If the user cannot use the service it's HIGH; if its related to billing/invoice set to Medium; everything else is Low. Data: "{}" """
    # Prompt for the Topic
    topic = """Given the text "{}", classify its topic into  Billing/Invoice|Technical/Bug Report|Account Management|Feature Request"""

    # Raplace this with your text array
    tickets = [
        """Subject: Cannot log in after the update! I changed my password 3 times but
    the system keeps throwing an error 500 when I hit 'Submit'. I have a demo with a client
    in 1 hour—this needs to be fixed immediately! I also noticed the new UI is missing the
    Dark Mode option, but that's secondary.""",
        """Subject: Will you make more formating options available in the future?""",
        """Subject: I need my invoice in order to make the payment"""
    ]

    # Create two prompts (urgency and topic) for each ticket in the list
    prompts = [prompt.format(ticket) for ticket in tickets for prompt in (urgency, topic)]

    # Query all at once - You shall break it down into smaller chunks for greater number of tickets
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=30,
        do_sample=False
    )

    controller = 0
    for t in range(len(tickets)):
        urgency_index = controller
        topic_index = controller + 1
        controller += 2

        urgency = tokenizer.decode(outputs[urgency_index], skip_special_tokens=True).strip()
        topic = tokenizer.decode(outputs[topic_index], skip_special_tokens=True).strip()

        print("Ticket id {} urgency: {}, topic {}".format(t, urgency, topic))


