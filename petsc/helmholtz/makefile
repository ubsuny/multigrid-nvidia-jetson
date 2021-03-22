all: helmholtz ex11

helmholtz: helmholtz.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}

ex11: ex11.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
