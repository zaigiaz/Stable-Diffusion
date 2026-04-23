neg_flags  := not ugly, not wierd, not disgusting, no blood, not bad, not abhorrent, not corrupted, not blurry
pos_flags  := masterpiece, accurate, great, realistic, elegant, clear lines, aesthetic
art_styles := cubist, impressionist, woodblock, futurist, album art

prompt_str := abstract, crystalline, futurist style, bizzare, strange, photorealistic, indigo silver color-scheme, album art, weird perspective, sharp edges

user_prompt := $(prompt_str) + $(pos_flags) + $(neg_flags)

target:
	python3 main.py --prompt "$(user_prompt)" -s "dpm++_2m"

json:
	python3 main.py -p "hello" -j "./data.json"

