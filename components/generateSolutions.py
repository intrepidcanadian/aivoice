def generateSolutions(co, problem, previous_solutions):
    # Format previous solutions for inclusion in the prompt

    # Construct a prompt that encourages generating a distinct solution
    prompt = (f"Given the problem: '{problem}', generate one possible creative solution. " +
              "Ensure the solution is distinct and leverages a different approach. " +
              f"Previous solutions include: {previous_solutions}. " +
              "Generate a new solution that is different from these.")

    response = co.generate(
        model='command-nightly',
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )

    # Extract and return the solution
    solution = response.generations[0].text.strip()
    return solution
