# HSI Image Filtering Practical

## Included files
- `hsi_filters.py` — Python program for HSI conversion and six spatial filters.
- `input/input.jpg` — sample colour image (replace it with your own image if desired).
- `output/` — generated results will be saved here when you run the program.
- `requirements.txt` — required Python libraries.

## Run in VS Code
1. Extract this ZIP file.
2. Open the extracted `HSI_Filter_Practical` folder in VS Code.
3. Open the VS Code terminal.
4. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Run:

   ```bash
   python hsi_filters.py
   ```

6. The comparison window will appear, and individual filtered images plus
   `all_filters_comparison.png` will be saved in the `output` folder.

## Use your own image
Replace `input/input.jpg` with your own colour image and keep the filename
`input.jpg`. The program uses a 3x3 neighbourhood for the spatial filters.

## Filters
- Minimum
- Median
- Maximum
- Averaging (3x3 mean)
- Weighted average (3x3 kernel)
- Gaussian (sigma = 1)

Note: This is an educational channel-wise HSI implementation. Hue is circular,
so a production-grade implementation should handle hue wraparound explicitly.
