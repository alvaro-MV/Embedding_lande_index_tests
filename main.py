import os
import pandas
import argparse
from roleplay import roleplay
from getpass import getpass
from expositive import expositive, expositive_mayeutic
from plot import plot_conversation, plot_expositive

def setup_api_key():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass("api key: ")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test for probing lande index as a proxy for information flow')
    parser.add_argument('-t', '--task',  dest ='task', 
                        action ='store', 
                        help ='task to be performed: <expositive> or <conversation>')
    return parser.parse_args()

def main():
    setup_api_key()
    args = parse_arguments()
    print(args.task)
    
    if (args.task == 'conversation'):
        resultado = roleplay()
        print(resultado)
        plot_conversation(resultado)

    elif (args.task == 'expositive'):
        resultado = expositive()
        resultado_df = pandas.DataFrame(resultado)
        print(resultado_df)
        plot_expositive(resultado)
        resultado_df.to_csv('data/tendencias_expositivo', sep=',')

    elif (args.task == 'expositive_mayeutic'):
        resultado = expositive_mayeutic()
        plot_conversation(resultado)

if __name__ == "__main__":
    main()
