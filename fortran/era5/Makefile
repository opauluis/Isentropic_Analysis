# Build the ERA5 isentropic analysis
#   make                 gfortran (default)
#   make FC=ifort        Intel (NCAR Casper: module load intel netcdf)
# netCDF-Fortran flags come from nf-config.

FC ?= gfortran
ifeq ($(FC),f77)
  FC = gfortran
endif
ifneq (,$(findstring gfortran,$(FC)))
  FFLAGS ?= -O2 -march=native
else
  FFLAGS ?= -O2 -xHost
endif
NC_FFLAGS := $(shell nf-config --fflags)
NC_FLIBS := $(shell nf-config --flibs)

OBJ = iso_thermo.o zonal_filter.o era5_iso.o

era5_iso.exe: $(OBJ)
	$(FC) $(FFLAGS) -o $@ $(OBJ) $(NC_FLIBS)

%.o: %.F90
	$(FC) $(FFLAGS) $(NC_FFLAGS) -c $<

era5_iso.o: iso_thermo.o zonal_filter.o

clean:
	rm -f *.o *.mod era5_iso.exe

.PHONY: clean
