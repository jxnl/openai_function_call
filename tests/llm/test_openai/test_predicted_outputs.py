import instructor

def test_predicted_outputs(client):
    client = instructor.patch(client, mode=instructor.Mode.TOOLS)

    # example from the openai docs for predicted outputs - https://platform.openai.com/docs/guides/predicted-outputs
    code = """
class User {
  firstName: string = "";
  lastName: string = "";
  username: string = "";
}

export default User;
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": 'Replace the "username" property with an "email" property. Respond only with code, and with no markdown formatting.',
            },
            {
                "role" : "user",
                "content" : code
            }
        ],
        prediction={
            "content": "string",
            "type": "content",
        },
    )
    assert response.usage.completion_tokens_details.accepted_prediction_tokens is not None
    assert response.usage.completion_tokens_details.rejected_prediction_tokens is not None
