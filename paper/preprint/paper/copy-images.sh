#!/bin/bash

echo "=== Starting image copy process ==="

# Create figures directory
mkdir -p figures
echo "Created/verified figures/ directory"

# Array to track filename mappings for later path updates
declare -A path_mapping

echo "Scanning .tex files for image references..."

# Find all .tex files and process them
for texfile in *.tex; do
    if [ -f "$texfile" ]; then
        echo "Processing: $texfile"
        
        # Extract image paths
        grep -o '\\includegraphics\[[^]]*\]{[^}]*}' "$texfile" | \
        sed 's/.*{\([^}]*\)}/\1/' | \
        while read -r img; do
            if [ -n "$img" ]; then
                echo "  Found reference: $img"
                
                if [ -f "$img" ]; then
                    # Get the basename and parent directory name
                    basename_img=$(basename "$img")
                    filename="${basename_img%.*}"
                    extension="${basename_img##*.}"
                    
                    # Create a unique name using parent directory path
                    # Replace / with _ and remove leading ../
                    parent_path=$(dirname "$img" | sed 's|^\.\./||g; s|\.\./||g; s|/|_|g')
                    
                    # Create unique filename: parentdir_originalname.ext
                    unique_name="${parent_path}_${basename_img}"
                    
                    # Copy with unique name
                    cp "$img" "figures/$unique_name"
                    echo "    ✓ Copied to: figures/$unique_name"
                    
                    # Store mapping for later
                    echo "$img|$unique_name" >> .image_mapping.tmp
                else
                    echo "    ✗ Warning: File not found: $img"
                fi
            fi
        done
    fi
done

# Now update all .tex files with the new paths
if [ -f .image_mapping.tmp ]; then
    echo ""
    echo "Updating image paths in .tex files..."
    
    for texfile in *.tex; do
        if [ -f "$texfile" ]; then
            echo "  Updating: $texfile"
            cp "$texfile" "${texfile}.bak"
            
            # Read each mapping and replace in tex file
            while IFS='|' read -r original_path new_filename; do
                # Escape special characters for sed
                escaped_original=$(echo "$original_path" | sed 's/[\/&]/\\&/g')
                escaped_new=$(echo "figures/$new_filename" | sed 's/[\/&]/\\&/g')
                
                # Replace the path in includegraphics (macOS requires empty string after -i)
                sed -i '' "s|{${escaped_original}}|{${escaped_new}}|g" "$texfile"
            done < .image_mapping.tmp
            
            echo "    ✓ Updated (backup saved as ${texfile}.bak)"
        fi
    done
    
    # Clean up temporary mapping file
    rm .image_mapping.tmp
fi

echo "=== Converting PNGs to PDFs for faster compilation ==="

for img in figures/*.png; do
    if [ -f "$img" ]; then
        pdf_file="${img%.png}.pdf"
        magick "$img" "$pdf_file"
        
        # Check if conversion succeeded
        if [ -f "$pdf_file" ]; then
            rm "$img"
            echo "  Converted: $(basename $img) → $(basename $pdf_file)"
        fi
    fi
done

# Update .tex file references
sed -i '' 's/\.png}/\.pdf}/g' *.tex

echo "Complete!"

# Count copied images
img_count=$(find figures/ -type f 2>/dev/null | wc -l)
echo ""
echo "=== Image copy process complete ==="
echo "Total images in figures/: $img_count"
echo "Original .tex files backed up with .bak extension"
