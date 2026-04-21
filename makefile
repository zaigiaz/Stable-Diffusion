neg_flags  := not ugly, not wierd, not disgusting, no blood, not bad, not abhorrent, not corrupted
pos_flags  := masterpiece, wonderful, great, realistic, pretty, appealing, elegant, clear lines, aesthetic, beautiful
art_styles := cubist, impressionist, woodblock, futurist, album art

prompt_str := create portrait of humanoid butterfly creature, surrealist painting, wierd perspective

user_prompt := $(prompt_str) + $(pos_flags) + $(neg_flags)

target:
	python3 main.py --prompt "$(user_prompt)" -s "dpm++_2m"

json:
	python3 main.py -p "hello" -j "./data.json"

