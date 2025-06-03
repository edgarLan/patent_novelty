import ast
import re

def quote_barewords_in_list(s):
    # Quote barewords inside brackets like [claims, background] -> ["claims", "background"]
    def replacer(match):
        word = match.group(0)
        if word.lower() in ['true', 'false']:
            return word
        if re.match(r'^-?\d+(\.\d+)?$', word):  # numeric
            return word
        return f'"{word}"'
    
    inner = s[1:-1]
    quoted = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', replacer, inner)
    return f'[{quoted}]'

def read_config_txt(filepath):
    config = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            key, value = line.strip().split("=", 1)
            value = value.strip()

            # Try to interpret list-like or nested structures
            if value.startswith("[") and value.endswith("]"):
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    try:
                        quoted = quote_barewords_in_list(value)
                        value = ast.literal_eval(quoted)
                    except Exception:
                        pass

            # Handle comma-separated values
            elif "," in value:
                value = [v.strip() for v in value.split(",") if v.strip()]
                new_list = []
                for v in value:
                    if v.lower() == "true":
                        new_list.append(True)
                    elif v.lower() == "false":
                        new_list.append(False)
                    elif v.isdigit():
                        new_list.append(int(v))
                    else:
                        try:
                            new_list.append(float(v))
                        except ValueError:
                            new_list.append(v)
                value = new_list
            else:
                # Handle single values
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]

            config[key] = value
    return config
