# Deep Q-Network Implementation using chainer

# Requirement

* [Chainer](http://chainer.org/)
* [PyAutoGUI](https://pyautogui.readthedocs.org/en/latest/)

# Supported Game

* [Winnie the Pooh's Home Run Derby](http://games.kids.yahoo.co.jp/sports/013.html)

# Usage

Run the game and:

```
python src/train.py -g 0 -l 537 -t 212 --width 600 --height 450 -o model\dqn --train_term 4 --random 0.4 --random_reduction 0.00002 --min_random 0.1
```

Options:
* -g, --gpu: (optional) GPU device index (default: -1).
* -l, --left: (required) left of the game screen in pixels.
* -t, --top: (required) top of the game screen in pixels.
* -i, --input: (optional) input model file path without extension.
* -o, --output: (required) output model file path without extension.
* -r, --random: randomness of playing (default: 0.2).
* --random_reduction: randomness reduction rate per iteration (default: 0.00002).
* --min_random: minimum randomness of playing (default: 0.1).

# License

MIT License
