import re

def split_text_into_groups(text, max_words=70):
    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    groups = []
    current_group = []
    current_word_count = 0
    
    for sentence in sentences:
        word_count = len(sentence.split())
        
        # Check if adding this sentence would exceed the max word count
        if current_word_count + word_count > max_words:
            # If the current group is not empty, add it to the groups list
            if current_group:
                groups.append(' '.join(current_group))
                current_group = []
                current_word_count = 0
            
            # If the sentence itself is too long, split it into smaller parts
            if word_count > max_words:
                words = sentence.split()
                while words:
                    part = ' '.join(words[:max_words])
                    groups.append(part)
                    words = words[max_words:]
            else:
                current_group.append(sentence)
                current_word_count += word_count
        else:
            current_group.append(sentence)
            current_word_count += word_count
    
    # Add any remaining group to the list
    if current_group:
        groups.append(' '.join(current_group))
    
    return groups

# Example usage
input_text = ("Yes, it's normal to find solutions to problems while falling asleep or waking up. "
              "This phenomenon is known as the \"hypnopompic state,\" which is a period of consciousness immediately following sleep. "
              "During this time, your mind is still in a somewhat dreamy, relaxed state, which can allow for insights and connections to form more easily.\n\n"
              "There are a few reasons why you might find solutions to problems during these periods:\n\n"
              "1. REM sleep: When you're falling asleep, your brain is transitioning from a state of alertness to a state of relaxation. "
              "This period, called the hypnagogic state, can be conducive to creative thinking and problem-solving, as your mind is less focused on the stresses and distractions of the day")


output_groups = split_text_into_groups(input_text)
for group in output_groups:
    print(group)
    print("----")
