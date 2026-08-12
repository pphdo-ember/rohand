# Rohand
A template to create executables from a Python script that uses [Mediapipe](https://pypi.org/project/mediapipe/).

# Workflow

## Local
Once you have a Python script, you can build an `.exe` (Windows) or an `.app` (Mac) file:
1. Create a virtual environment and activate it.
2. Run: `pip install requirements.txt`.
3. Download `main.spec` and `hand_landmarker.task`. If you have another model you want to use, open up `main.spec` and change `datas = [('hand_landmarker.task'), '.')]` to use your `.task` model.
4. Run: `pyinstaller --clean main.spec`. Note that `main.spec` looks for a file named `main.py`; you can change that in the `Analysis` constructor.
5. Go to `./dist` and double click on the `.exe` or `.app` file there.

## Building for other machines
[PyInstaller](https://pypi.org/project/pyinstaller/) creates an executable for the machine it's running on. This means that if you're on Windows and you want to distribute for Mac users, you'll need a Mac machine to run `pyinstaller --clean main.spec` on.

One way to build executables for other machines is to use GitHub Actions:
1. Download the `requirements.txt` file.
1. Create a GitHub repository with your 4 files: `main.py`, `hand_landmarker.task`, `main.spec`, `requirements.txt`.
2. Go to Actions > New workflow > Set up a workflow yourself. This creates a blank `main.yml` file.
3. Copy the text in this repository (rohand's) `.github/workflows/main.yml` and paste it into yours. Commit.
4. Then, run it. The artifacts will appear at the bottom once it's done.
