from sentence_transformers import SentenceTransformer, util

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined FAQ knowledge base (can be expanded)
faq_data = {
    "What is the eligibility criteria for placements?":
        "Students must have at least 60% aggregate and no current backlogs.",
    "What is the last date to register for placements?":
        "The last date to register is 30th September.",
    "Can students with backlogs sit for placements?":
        "No, students with current backlogs are not eligible to sit for placements.",
    "How many companies can a student apply to?":
        "Students can apply to any number of companies unless restricted by the company.",
    "Are internships mandatory for placements?":
        "Internships are not mandatory but are highly recommended.",
    "How can I prepare for technical interviews?":
        "You can prepare using platforms like LeetCode, GeeksforGeeks, and practicing your resume projects.",
    "What documents are needed during placement?":
        "Carry your updated resume, college ID, and academic transcripts."
}

# Convert questions to embeddings
faq_questions = list(faq_data.keys())
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)

# Chatbot loop
print("🤖 Placement Chatbot Assistant (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Bot: Thank you! All the best for your placements.")
        break

    # Encode user query
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    # Compute similarity with FAQ questions
    similarities = util.cos_sim(query_embedding, faq_embeddings)[0]

    # Find best matching FAQ
    best_match_idx = similarities.argmax().item()
    best_score = similarities[best_match_idx].item()

    # Threshold to determine if answer is confident
    if best_score > 0.6:
        answer = faq_data[faq_questions[best_match_idx]]
        print(f"Bot: {answer}\n")
    else:
        print("Bot: I'm not sure about that. Please contact the placement cell for more details.\n")


#EXAMPLE 

🤖 Placement Chatbot Assistant (type 'exit' to quit)

You: What is the deadline to register?
Bot: The last date to register is 30th September.

You: Do I need an internship?
Bot: Internships are not mandatory but are highly recommended.

You: How to prepare for coding rounds?
Bot: You can prepare using platforms like LeetCode, GeeksforGeeks, and practicing your resume projects.

You: exit
Bot: Thank you! All the best for your placements.


#LIBRARY 

#pip install sentence-transformers
