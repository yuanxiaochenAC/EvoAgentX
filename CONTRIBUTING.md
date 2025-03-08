# Contributing to EvoAgentX

Thank you for considering contributing to **EvoAgentX** – an automatic agentic workflow generation and evolving framework! We appreciate your interest and contributions to improving the project.

## 🚀 Getting Started

1. **Fork the Repository**: Click the 'Fork' button on the top right of this repo to create your own copy.
2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/your-username/EvoAgentX.git
   ```
3. **Navigate to the Project Directory**:
   ```bash
   cd EvoAgentX
   ```
4. **Create a Virtual Environment**:
   ```bash
   conda create -n agent python=3.10
   conda activate agent
   ```
5. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 How to Contribute

### **1. Reporting Bugs** 🐞
- Ensure the bug has not been reported in the [Issues](https://github.com/EvoAgentX/issues) tab.
- Provide detailed steps to reproduce the issue.
- Include logs, screenshots, and environment details.
- Use a clear title and concise description.

### **2. Suggesting Features** 💡
- Check if a similar feature request exists in [Issues](https://github.com/EvoAgentX/issues).
- Provide a use case explaining why this feature is beneficial.
- Suggest a possible implementation approach if possible.

### **3. Code Contributions** 👨‍💻
#### **Step 1: Create a Branch**
Before making changes, create a new branch:
```bash
git checkout -b feature/your-feature-name
```

#### **Step 2: Implement Your Changes**
- Follow the project’s coding style.
- Write clean, maintainable, and well-documented code.
- Add relevant unit tests for your changes.

#### **Step 3: Commit Your Changes**
- Follow conventional commit messages:
  ```bash
  git commit -m "feat: add new agent workflow module"
  ```
- Use descriptive commit messages.

#### **Step 4: Push Your Changes**
```bash
git push origin feature/your-feature-name
```

#### **Step 5: Submit a Pull Request**
- Navigate to the main repository and create a **Pull Request** (PR).
- Provide a clear description of your changes.
- Reference the related issue if applicable.
- Wait for a review and respond to feedback promptly.

## 📏 Coding Guidelines

All Python functions, classes, and modules in this project should use **NumPy-style docstrings**. Below are the guidelines and examples that contributors must follow to maintain consistency and clarity throughout the codebase.

---


## Basic Structure

A typical NumPy-style docstring contains:

1. **Short Summary**  
   A one-line (or short paragraph) summary describing what the function or class does.
2. **Extended Description** (optional)  
   A longer explanation or background details.
3. **Parameters**  
   A list of parameters with their types and descriptions.
4. **Returns** (or **Yields** if it’s a generator)  
   The return type(s) and explanation.
5. **Raises** (optional)  
   Document the exceptions that might be raised.
6. **Additional Sections** (optional)  
   Such as **Notes**, **Warns**, **Examples**, etc.

**Example Layout**:

```python
def function_name(param1, param2):
    """
    Brief summary of the function.

    Detailed explanation or reasoning (optional). Can be multiple lines.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str
        Description of param2. For long text, continue on a new line
        and indent properly.

    Returns
    -------
    bool
        Explanation of the return value.

    Raises
    ------
    ValueError
        Explanation of when/why this error is raised.
    """
    # function body
    pass

class Calculator:
    """
    A simple calculator class to demonstrate NumPy-style docstrings.

    This class provides basic arithmetic operations. Advanced features can be
    added as needed.

    Attributes
    ----------
    last_result : float
        Stores the result of the most recent operation.

    Methods
    -------
    add(a, b)
        Returns the sum of two numbers.
    subtract(a, b)
        Returns the difference between two numbers.
    """

    def __init__(self):
        """
        Initialize the Calculator with a default result of 0.
        """
        self.last_result = 0

    def add(self, a, b):
        """
        Add two numbers and update `last_result`.

        Parameters
        ----------
        a : float
            The first number.
        b : float
            The second number.

        Returns
        -------
        float
            The sum of a and b.
        """
        self.last_result = a + b
        return self.last_result

    def subtract(self, a, b):
        """
        Subtract b from a and update `last_result`.

        Parameters
        ----------
        a : float
            The number to be subtracted from.
        b : float
            The number to subtract.

        Returns
        -------
        float
            The result of a - b.
        """
        self.last_result = a - b
        return self.last_result
```

## 🧪 Testing
Before submitting a PR, ensure that all tests pass:
```bash
pytest tests/
```

## 🔄 Syncing Your Fork
To stay updated with the latest changes:
```bash
git fetch upstream
```
```bash
git merge upstream/main
```

## 🤝 Community Guidelines
- Be respectful and inclusive.
- Provide constructive feedback.
- Keep discussions relevant to the project.

## 📩 Contact
For any questions, feel free to open an issue or reach out to the maintainers.

Happy coding! 🚀
