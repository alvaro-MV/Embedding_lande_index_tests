import os
import argparse
from roleplay import roleplay
from getpass import getpass
from expositive import expositive

os.environ["OPENAI_API_KEY"] = getpass("api key: ")
parser = argparse.ArgumentParser(description = 'Test for probing lande index as a proxy for information flow')
 
parser.add_argument('-t', '--task',  dest ='task', 
                    action ='store', 
                    help ='task to be performed: <expositive> or <conversation>')

args = parser.parse_args()
print(args.task)

if (args.task == 'conversation'):
	result = roleplay()
elif (args.task == 'expositive'):
	result = expositive()
