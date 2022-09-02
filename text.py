import pandas as pd
from transformers import pipeline
MIN_LENGTH, MAX_LENGTH = 200, 500
generator = pipeline('text-generation', model="facebook/opt-1.3b")
with open('prompts.txt', 'r') as f:
	prompts = f.readlines()
	generator_results = generator(prompts, min_length=MIN_LENGTH, max_length=MAX_LENGTH)
	generated_text = [l[0]['generated_text'][len(p):] for p, l in zip(prompts, generator_results)]
	results = pd.DataFrame({'Prompt':prompts, 'Generated Text':generated_text})
	results.to_csv('results.csv')
	for i in range(len(results)):
		print('###########\nPrompt:', results.loc[i, 'Prompt'])
		print('\nGenerated text:\n###########\n', results.loc[i, 'Generated Text'], '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
