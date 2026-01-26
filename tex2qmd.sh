# Tex to Quarto function:
tex2qmd() {
    # Check if input file is provided
    if [ $# -eq 0 ]; then
        echo "Usage: tex2qmd <input.tex> [output.qmd]"
        return 1
    fi

    # Input file (first argument)
    local input_file="$1"

    # Determine output file
    local output_file
    if [ $# -eq 2 ]; then
        # Use provided output file name
        output_file="$2"
    else
        # Generate output file name by replacing .tex with .qmd
        output_file="${input_file%.tex}.qmd"
    fi

    # Run pandoc conversion
    pandoc -s \
        --wrap=none \
        -t markdown \
        -f latex \
        --lua-filter=fix_section_refs.lua \
        "$input_file" \
        -o "$output_file"

    # Print confirmation
    echo "Converted $input_file to $output_file"
}

