## Setting up the dataset (PigLife)

1. Request access to the dataset at [PigLife - AIFARMS Data Portal](https://data.aifarms.org/view/piglife).
2. After receiving the private download URL, create your environment file:
	- Copy `.env.example` to `.env`
	- Set `PIGLIFE_URL` with your download link
3. Check disk space before downloading and extracting the data:
	- Recommended free space: at least **20 GB** (dataset is about **16 GB**)
4. Install dependencies:
	- `uv sync`
5. Run the complete dataset pipeline:
	- `uv run inv build`