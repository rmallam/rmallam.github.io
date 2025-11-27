#!/bin/bash

# Generate blog index JSON
echo '{' > blogs/index.json
first_cat=true

for dir in blogs/*/; do
    if [ -d "$dir" ]; then
        category=$(basename "$dir")
        
        if [ "$first_cat" = true ]; then
            first_cat=false
        else
            echo ',' >> blogs/index.json
        fi
        
        echo "  \"$category\": [" >> blogs/index.json
        
        first_file=true
        for file in "$dir"*.md; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                
                if [ "$first_file" = true ]; then
                    first_file=false
                else
                    echo ',' >> blogs/index.json
                fi
                
                echo -n "    \"$filename\"" >> blogs/index.json
            fi
        done
        
        echo '' >> blogs/index.json
        echo -n '  ]' >> blogs/index.json
    fi
done

echo '' >> blogs/index.json
echo '}' >> blogs/index.json

echo "Blog index generated at blogs/index.json"
