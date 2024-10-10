"""
LLM prompts
'General Explanation' explains the task to the LLM -- the context of what it's doing
as well as the rules for response. 
'Provide History' asks the LLM to look at its previous attempts and
generate a new response based on the data.
'Reminder' reminds the LLM of the formatting rules and goals of the task, along
with any other important information we want to add.
"""

def get_prompt_backbone(style):
    prompts = {
        "prompt_style_1": {
            "general_explanation": f""" You are overseeing a MARL environment... """,
            "provide_history": """ A history of your previous attempts tells you...  Here is your history of attempts: """,
            "reminder": """ Please remember that the tax rate that you generate should be in the form... """,
        },
        "prompt_style_2": {
            "general_explanation": f""" You are overseeing a MARL environment... """,
            "provide_history": """ A history of your previous attempts tells you...  Here is your history of attempts: """,
            "reminder": """ Please remember that the tax rate that you generate should be in the form... """,
        },
    }

    return prompts[style]
