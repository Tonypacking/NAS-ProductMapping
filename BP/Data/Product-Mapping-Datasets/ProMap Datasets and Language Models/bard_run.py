import pandas as pd
import bardapi
import time 
token = '' # Google Bard token from cookies goes here

bard_model =  bardapi.core.Bard(token)
input_file = 'bard/promapcz_names_bard.csv'
output_file = 'results.txt'
language = 'en' # cz en
responses = []

df = pd.read_csv(input_file, sep='\t', encoding='utf-8')
data = df[['name1', 'name2']].values
for i in range(0,len(data)):
    if language=='cz':
        input_text = f"Jsou tyto dva produkty stejné? Napiš 1 pokud ano a 0 pokud ne. \n PRVNI PRODUKT: {data[i][0]} \n DRUHY PRODUKT: {data[i][1]}"
    else:
        input_text = f"Are these two products the same? Print 1 if yes or 0 if no. \n FIRST PRODUCT: {data[i][0]} \n SECOND PRODUCT: {data[i][1]}"
    response = bard_model.get_answer(input_text)
    responses.append(str(i+2) + ' ' + response['content'] + '\n--------')
    print(str(i+2) + ' ' + response['content'] + '\n--------')
    time.sleep(1)

with open(output_file, 'w', encoding='utf-8') as f:
    for i in responses:
        f.write(i)