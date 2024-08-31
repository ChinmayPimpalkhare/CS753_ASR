import json
import unicodedata


def convert_to_devanagari(text):
    return unicodedata.normalize('NFKD', text).encode('utf-8', 'ignore').decode('utf-8')

def convert_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for entry in data:
        if 'ground_truth' in entry:
            entry['ground_truth'] = convert_to_devanagari(entry['ground_truth'])
        if 'prediction' in entry:
            entry['prediction'] = convert_to_devanagari(entry['prediction'])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    input_file1 = 'fine_tuned_whisper_with_constrained_beam_search.json'
    output_file1 = 'converted_fine_tuned_whisper_with_constrained_beam_search.json'
    convert_json(input_file1, output_file1)
    input_file2 = 'finetuned_zero_shot_whisper.json'
    output_file2 = 'converted_finetuned_zero_shot_whisper.json'
    convert_json(input_file2, output_file2)
    input_file3 = 'zero_shot_whisper.json'
    output_file3 = 'converted_zero_shot_whisper.json'
    convert_json(input_file3, output_file3)
    input_file4 = 'fine_tuned_whisper_with_constrained_beam_search-2.json'
    output_file4 = 'converted_fine_tuned_whisper_with_constrained_beam_search-2.json'
    convert_json(input_file4, output_file4)
    input_file5 = 'fine_tuned_whisper_with_constrained_beam_search-3.json'
    output_file5 = 'converted_fine_tuned_whisper_with_constrained_beam_search-3.json'
    convert_json(input_file5, output_file5)
    input_file6 = 'fine_tuned_whisper_with_beam_search.json'
    output_file6 = 'converted_fine_tuned_whisper_with_beam_search.json'
    convert_json(input_file6, output_file6)