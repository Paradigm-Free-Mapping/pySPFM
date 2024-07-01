{mod}`{{module}}`.{\{objname}}
{{ underline }}==============

```{eval-rst}
.. currentmodule:: {{ module }}
```

```{eval-rst}
.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:
   {% block methods %}
   {% endblock %}
```

```{eval-rst}
.. include:: {{fullname}}.examples
```

```{raw} html
<div class="clearer"></div>
```
