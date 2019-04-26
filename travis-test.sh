#! /bin/bash
if [[ $BUILDDOCS == 1 ]]; then
  cd docs && make html
else
  pytest --pyargs konrad
fi
