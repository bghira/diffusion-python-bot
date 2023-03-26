import os
import inspect


# Register an autoloader for classes
def autoload(classname):
    classpath = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'classes', classname + '.php')
    if os.path.exists(classpath):
        with open(classpath, 'r') as f:
            exec(f.read())


# Initialize DirectoryIterator for the classes directory
directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'classes')
output_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'class_signatures.txt')

with open(output_file, 'w') as f:
    f.write('') # Clear output file

for filename in os.listdir(directory):
    if filename.endswith('.php'):
        classname = os.path.splitext(filename)[0]
        with open(os.path.join(directory, filename), 'r') as f:
            module = type('', (), {})()
            exec(f.read(), module.__dict__)

            class_signatures = f"Class: {classname}\n"
            for method_name, method in inspect.getmembers(module, predicate=inspect.isfunction):
                if method.__module__ == classname:
                    parameters = inspect.signature(method).parameters
                    param_strings = []
                    for param_name, param in parameters.items():
                        param_type = param.annotation.__name__ + ' ' if param.annotation != inspect.Parameter.empty else ''
                        param_strings.append(f"{param_type}${param_name}")

                    param_list = ', '.join(param_strings)
                    return_type = f": {method.__annotations__['return'].__name__}" if 'return' in method.__annotations__ else ''

                    class_signatures += f"{method_name}({param_list}){return_type}\n"

                    # Extract example return values from the method's docblock
                    docstring = method.__doc__
                    if docstring is not None:
                        match = re.search(r'@return.*\{(.*?)\}', docstring, re.DOTALL)
                        if match is not None:
                            example_return_value = match.group(1).strip()
                            class_signatures += f"  Example Return Value: {example_return_value}\n"

            # Append class signatures to the output file
            with open(output_file, 'a') as f:
                f.write(class_signatures + '\n')