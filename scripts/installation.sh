#!/bin/bash
pip install venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Change weights_only to False in torch/serialization.py
SERIALIZATION_FILE="venv/lib/python3.12/site-packages/torch/serialization.py"

# Check if the file exists
if [ -f "$SERIALIZATION_FILE" ]; then
    echo "Modifying $SERIALIZATION_FILE..."
    
    # Use sed to replace the line containing weights_only with weights_only = False
    # This handles both the parameter declaration and the default value assignment
    sed -i 's/weights_only: Optional\[bool\] = None,/weights_only: Optional\[bool\] = False,/g' "$SERIALIZATION_FILE"
    sed -i 's/weights_only = None/weights_only = False/g' "$SERIALIZATION_FILE"
    
    echo "Successfully modified weights_only to False in $SERIALIZATION_FILE"
else
    echo "Error: $SERIALIZATION_FILE not found. Check the Python version or path."
    
    # Try to find the correct path
    ALTERNATIVE_PATHS=$(find venv/lib -name "serialization.py" | grep torch)
    if [ ! -z "$ALTERNATIVE_PATHS" ]; then
        echo "Found alternative path(s):"
        echo "$ALTERNATIVE_PATHS"
        echo "Please update the script with the correct path."
    fi
fi