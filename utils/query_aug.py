import os
import json
import argparse
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def paraphrase_text(text):
    prompt = (
        f"Please paraphrase the following text in clear English, "
        f"keeping the meaning as close as possible to the original:\n\n{text}"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert in short audio event descriptions."},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )
    return response.choices[0].message.content.strip()

def paraphrase_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        entry['global_caption'] = paraphrase_text(entry['global_caption'])
        for moment in entry['moments']:
            moment['local_caption'] = paraphrase_text(moment['local_caption'])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Paraphrase captions in CASTELLA JSON')
    parser.add_argument('input', help='Путь к входному JSON-файлу')
    parser.add_argument('output', help='Путь к выходному JSON-файлу')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Ошибка: входной файл '{args.input}' не найден.")
        exit(1)

    paraphrase_json(args.input, args.output)
    print(f"Готово! Результат сохранён в {args.output}")