// Js script to make the Jupyter notebook recognize tabs on the cells

%%javascript
IPython.tab_as_tab_everywhere = function(use_tabs) {
    if (use_tabs === undefined) {
        use_tabs = true;
    }

    // apply setting to all current CodeMirror instances
    IPython.notebook.get_cells().map(
            function(c) {  return c.code_mirror.options.indentWithTabs=use_tabs;  }
    );
    // make sure new CodeMirror instances created in the future also use this setting
    CodeMirror.defaults.indentWithTabs=use_tabs;

};

IPython.tab_as_tab_everywhere()
