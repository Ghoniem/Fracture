"""
FunctionLocator - Python Function Parameter Analysis Tool
==========================================================

A utility class for analyzing function parameters, finding valid options,
and exploring function signatures in Jupyter notebooks.

Author: Auto-generated
License: MIT
"""

import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class FunctionLocator:
    """
    Analyze function parameters and find valid options.
    
    This class provides methods to:
    - Find valid options for string parameters
    - Show all parameters with types and defaults
    - Search for classes and functions in modules
    - Display function source code
    
    Examples
    --------
    >>> from function_locator import FunctionLocator
    >>> 
    >>> # Analyze a function
    >>> loc = FunctionLocator(calc.solve)
    >>> 
    >>> # Find valid options for a parameter
    >>> loc.what_options('parametrization')
    >>> 
    >>> # Show all parameters
    >>> loc.show_params()
    >>> 
    >>> # Get source code
    >>> loc.show_source()
    """
    
    def __init__(self, func: Optional[Callable] = None):
        """
        Initialize FunctionLocator.
        
        Parameters
        ----------
        func : callable, optional
            The function to analyze. Can be set later with set_function().
        """
        self.func = func
        self._source_cache = None
    
    def set_function(self, func: Callable):
        """
        Set or change the function to analyze.
        
        Parameters
        ----------
        func : callable
            The function to analyze
        """
        self.func = func
        self._source_cache = None
        return self
    
    def _get_source(self) -> str:
        """Get source code (cached)."""
        if self._source_cache is None:
            if self.func is None:
                raise ValueError("No function set. Use set_function() first.")
            try:
                self._source_cache = inspect.getsource(self.func)
            except Exception as e:
                raise RuntimeError(f"Could not get source code: {e}")
        return self._source_cache
    
    def what_options(self, param_name: str, verbose: bool = True) -> List[str]:
        """
        Find valid options for a parameter.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter to analyze
        verbose : bool, default=True
            If True, print detailed output. If False, just return the list.
        
        Returns
        -------
        List[str]
            List of possible valid values found in source code
        
        Examples
        --------
        >>> loc = FunctionLocator(calc.solve)
        >>> loc.what_options('parametrization')
        >>> options = loc.what_options('crack_mode', verbose=False)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"VALID OPTIONS FOR: {param_name}")
            print('='*60)
        
        source = self._get_source()
        
        # Find all lines mentioning this parameter
        lines = source.split('\n')
        relevant_lines = []
        
        for i, line in enumerate(lines, 1):
            if param_name in line and not line.strip().startswith('#'):
                relevant_lines.append((i, line.strip()))
        
        if not relevant_lines:
            if verbose:
                print(f"⚠️  Parameter '{param_name}' not used in function body")
            return []
        
        # Extract quoted values
        all_values = set()
        for _, line in relevant_lines:
            # Match single and double quoted strings
            values = re.findall(r'["\']([a-zA-Z0-9_]+)["\']', line)
            all_values.update(values)
        
        if verbose:
            print(f"\n📋 Found {len(relevant_lines)} reference(s) in code:")
            for line_num, line in relevant_lines[:5]:  # Show first 5
                print(f"   Line {line_num}: {line}")
            
            if all_values:
                print(f"\n✓ Possible valid values:")
                for val in sorted(all_values):
                    print(f"   • {val}")
            else:
                print(f"\n⚠️  No explicit values found")
                print("   Try: loc.show_source() to see full code")
        
        return sorted(list(all_values))
    
    def show_params(self, filter_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Show all parameters with their types and defaults.
        
        Parameters
        ----------
        filter_type : str, optional
            If provided, only show parameters of this type (e.g., 'str', 'int')
        
        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to their info
        
        Examples
        --------
        >>> loc = FunctionLocator(calc.solve)
        >>> loc.show_params()
        >>> loc.show_params(filter_type='str')  # Only string parameters
        """
        if self.func is None:
            raise ValueError("No function set. Use set_function() first.")
        
        sig = inspect.signature(self.func)
        
        print(f"\n{'='*60}")
        print(f"ALL PARAMETERS FOR: {self.func.__name__}")
        print('='*60)
        
        params_info = {}
        
        for name, param in sig.parameters.items():
            if name in ['self', '_ignored']:
                continue
            
            param_type = str(param.annotation).replace("'", "")
            default = param.default if param.default != inspect.Parameter.empty else "REQUIRED"
            
            # Filter by type if requested
            if filter_type and param_type != filter_type:
                continue
            
            print(f"\n{name}:")
            print(f"  Type    : {param_type}")
            print(f"  Default : {default}")
            
            params_info[name] = {
                'type': param_type,
                'default': default,
                'annotation': param.annotation
            }
        
        return params_info
    
    def show_source(self, lines: Optional[Tuple[int, int]] = None):
        """
        Display function source code.
        
        Parameters
        ----------
        lines : tuple of (start, end), optional
            Show only specific line range. If None, show all.
        
        Examples
        --------
        >>> loc = FunctionLocator(calc.solve)
        >>> loc.show_source()
        >>> loc.show_source(lines=(1, 20))  # First 20 lines
        """
        source = self._get_source()
        
        print(f"\n{'='*60}")
        print(f"SOURCE CODE FOR: {self.func.__name__}")
        print('='*60)
        
        source_lines = source.split('\n')
        
        if lines:
            start, end = lines
            source_lines = source_lines[start-1:end]
            print(f"(Showing lines {start}-{end})")
        
        for i, line in enumerate(source_lines, 1):
            print(f"{i:4}: {line}")
    
    def search_source(self, keyword: str, context: int = 2):
        """
        Search for a keyword in source code and show context.
        
        Parameters
        ----------
        keyword : str
            Keyword to search for
        context : int, default=2
            Number of lines to show before/after match
        
        Examples
        --------
        >>> loc = FunctionLocator(calc.solve)
        >>> loc.search_source('parametrization', context=3)
        """
        source = self._get_source()
        lines = source.split('\n')
        
        print(f"\n{'='*60}")
        print(f"SEARCHING FOR: '{keyword}'")
        print('='*60)
        
        matches = []
        for i, line in enumerate(lines):
            if keyword in line:
                matches.append(i)
        
        if not matches:
            print(f"\n⚠️  '{keyword}' not found in source code")
            return
        
        print(f"\n✓ Found {len(matches)} occurrence(s):\n")
        
        for match_idx in matches:
            start = max(0, match_idx - context)
            end = min(len(lines), match_idx + context + 1)
            
            print(f"Line {match_idx + 1}:")
            print("-" * 40)
            for i in range(start, end):
                marker = ">>>" if i == match_idx else "   "
                print(f"{marker} {i+1:4}: {lines[i]}")
            print()
    
    def get_signature(self) -> inspect.Signature:
        """
        Get the function signature.
        
        Returns
        -------
        inspect.Signature
            The function signature object
        
        Examples
        --------
        >>> loc = FunctionLocator(calc.solve)
        >>> sig = loc.get_signature()
        >>> print(sig)
        """
        if self.func is None:
            raise ValueError("No function set. Use set_function() first.")
        return inspect.signature(self.func)
    
    def find_all_string_params(self) -> Dict[str, List[str]]:
        """
        Find all string parameters and their possible values.
        
        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping string parameter names to possible values
        
        Examples
        --------
        >>> loc = FunctionLocator(calc.solve)
        >>> options = loc.find_all_string_params()
        >>> for param, values in options.items():
        ...     print(f"{param}: {values}")
        """
        sig = self.get_signature()
        
        results = {}
        for param_name, param in sig.parameters.items():
            if param_name in ['self', '_ignored']:
                continue
            
            param_type = str(param.annotation).replace("'", "")
            if param_type == 'str':
                options = self.what_options(param_name, verbose=False)
                if options:
                    results[param_name] = options
        
        return results
    
    @staticmethod
    def find_class(class_name: str, module_path: str):
        """
        Find a class in a module or package.
        
        Parameters
        ----------
        class_name : str
            Name of the class to find
        module_path : str
            Path to search (e.g., 'fracture_utils')
        
        Examples
        --------
        >>> FunctionLocator.find_class('Material', 'fracture_utils')
        """
        from pathlib import Path
        
        root = Path(module_path)
        
        print(f"\n{'='*60}")
        print(f"SEARCHING FOR CLASS: {class_name}")
        print('='*60)
        print(f"In: {root}\n")
        
        for py_file in root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if f'class {class_name}' in line:
                            rel_path = py_file.relative_to(root.parent)
                            print(f"✓ Found in: {rel_path}")
                            print(f"  Line {i}: {line.strip()}")
                            
                            # Show import statement
                            import_path = str(rel_path).replace('\\', '.').replace('/', '.').replace('.py', '')
                            print(f"\n  Import:")
                            print(f"    from {import_path} import {class_name}")
                            return
            except Exception:
                pass
        
        print(f"❌ Class '{class_name}' not found")
    
    def __repr__(self):
        func_name = self.func.__name__ if self.func else "None"
        return f"FunctionLocator(func={func_name})"


# Convenience function for quick use
def analyze(func: Callable, param: Optional[str] = None):
    """
    Quick analysis of a function.
    
    Parameters
    ----------
    func : callable
        Function to analyze
    param : str, optional
        If provided, show options for this parameter only
    
    Examples
    --------
    >>> analyze(calc.solve)  # Show all parameters
    >>> analyze(calc.solve, 'parametrization')  # Show options for one param
    """
    loc = FunctionLocator(func)
    
    if param:
        loc.what_options(param)
    else:
        loc.show_params()
