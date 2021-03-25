all: helmholtz

helmholtz: helmholtz.o
	${CLINKER} -o $@ $^ ${PETSC_LIB}


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
