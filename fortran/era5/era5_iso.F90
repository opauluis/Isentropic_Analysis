! Daily isentropic analysis of ERA5 on pressure levels (theta_e coordinates)
! Olivier Pauluis
!
! Reads the 24 hourly ERA5 analyses of one day (RDA d633000, 0.25 deg), computes theta_e, moist
! entropy and enthalpy, splits omega into its zonal mean, zonal wavenumbers 1-3, 4-10 and the
! remainder, and bins mass flux and tracers in theta_e on 3x3 degree boxes. Points below the
! surface (Z < Z_sfc) are left out. Output is the daily mean of the conditional sums and the
! isentropic streamfunctions STR*_THE(theta_e, p) = cumulative sum over theta_e of the mass flux.
!
! Usage:  era5_iso YEAR MONTH DAY [OUT_DIR]
!   writes OUT_DIR/YYYY/ERA5_STR_THE.YYYY_MM_DD.nc
!   ERA5_PL_DIR and ERA5_ZS_FILE (environment) override the input locations on Casper.
!
! Computations are in double precision; the output is stored in single precision.
! Mass fluxes are omega * area / g (kg s-1), positive downward. Box coordinates follow the ERA5 order, so latitude
! runs from north to south.

program era5_iso
  use netcdf
  use iso_thermo, only: moist_thermo
  use zonal_filter, only: band_pass
  implicit none

  integer, parameter :: dp = kind(1.0d0)

  character(len=*), parameter :: DIR_ERA5_DEFAULT = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
  character(len=*), parameter :: FILE_ZS_DEFAULT = "/glade/campaign/collections/rda/data/d633000/" // &
       "e5.oper.invariant/197901/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
  character(len=*), parameter :: DIR_OUT_DEFAULT = "/glade/derecho/scratch/pauluis/ERA5/ISO/THETAE/"
  character(len=*), parameter :: FILE_OUT = "ERA5_STR_THE."

  ! theta_e bins
  integer, parameter :: NBIN = 181
  real(dp), parameter :: THE_MIN = 200.0_dp, THE_MAX = 380.0_dp
  ! output boxes: 12 longitudes, 13 latitudes with the edge rows shared by neighbouring boxes
  integer, parameter :: nlon_slice = 12, nlat_slice = 13
  ! top pressure level (hPa)
  real(dp), parameter :: P_TOP = 150.0_dp

  real(dp), parameter :: radius = 6378e3_dp
  real(dp), parameter :: grav = 9.81_dp

  ! input fields, in the order of the files
  integer, parameter :: NVAR = 8
  integer, parameter :: iW = 1, iT = 2, iZ = 3, iQ = 4, iCIWC = 5, iCLWC = 6, iCSWC = 7, iCRWC = 8
  character(len=8), parameter :: VAR_NAME(NVAR) = &
       [character(len=8) :: "W", "T", "Z", "Q", "CIWC", "CLWC", "CSWC", "CRWC"]
  character(len=8), parameter :: VAR_CODE(NVAR) = &
       [character(len=8) :: "135_w", "130_t", "129_z", "133_q", "247_ciwc", "246_clwc", "076_cswc", "075_crwc"]

  ! binned fields on (theta_e, lon, lat, press)
  integer, parameter :: NF = 12
  integer, parameter :: fAREA = 1, fMF = 2, fT = 3
  character(len=10), parameter :: F_NAME(NF) = [character(len=10) :: "AREA_THE", "MFC_THE", "RHOT_THE", &
       "RHOQ_THE", "RHOZ_THE", "RHOS_THE", "RHOH_THE", "RHOQL_THE", "RHOQI_THE", "RHOWQ_THE", &
       "RHOWQL_THE", "RHOWQI_THE"]
  ! binned omega components on (theta_e, lat, press)
  integer, parameter :: NFZ = 4
  character(len=10), parameter :: FZ_NAME(NFZ) = [character(len=10) :: "MF0_THE", "MF13_THE", &
       "MF410_THE", "MFSW_THE"]
  character(len=10), parameter :: STR_NAME(NFZ) = [character(len=10) :: "STR0_THE", "STR13_THE", &
       "STR410_THE", "STRSW_THE"]

  character(len=256) :: dir_era5, file_zs, dir_out, file_name, arg
  character(len=4) :: yearname
  character(len=2) :: monthname, dayname
  integer :: year, month, day

  integer :: NLON, NLAT, NLEV, NX, NY, ntime, nlev_file, level_start
  real(dp), allocatable :: lat(:), lon(:), pressure(:), press(:), lat_out(:), lon_out(:), the_lev(:)
  real(dp), allocatable :: area_row(:)
  real(dp), allocatable :: temp(:,:,:), sphum(:,:,:), omega(:,:,:), qliq(:,:,:), qice(:,:,:), GZ(:,:,:), GZS(:,:)
  real(dp), allocatable :: dummy(:,:,:), entropy(:,:,:), enthalpy(:,:,:), theta_e(:,:,:)
  real(dp), allocatable :: O0(:,:,:), O13(:,:,:), O410(:,:,:), OSW(:,:,:)
  logical, allocatable :: above(:)
  real(dp), allocatable :: ACC(:,:,:,:,:), ACCZ(:,:,:,:)
  real(dp), allocatable :: MFC_THE(:,:,:,:), MFCNV_THE(:,:,:,:), AREA_Y(:,:,:)
  real(dp), allocatable :: STR_THE(:,:), STRZ_THE(:,:,:)
  real(dp), allocatable :: WNET(:), WLAT(:,:), WCOL(:,:,:)

  integer :: ncid(NVAR), varid(NVAR), ncid_zs, varid_zs, varid_c
  integer :: ncid_out, x_dimid, y_dimid, z_dimid, the_dimid, rec_dimid
  integer :: varid_out(NF + 1), varid_z(NFZ), varid_str, varid_strz(NFZ), varid_coord(4)
  integer :: istep, i, j, k, l, n, ibin, jbin, isl, jsl, bin_ind, b
  real(dp) :: x, weight, w(2), vals(NF), valz(NFZ), weight_slc(nlat_slice), dthe
  real(dp) :: dlat, pi

  pi = 4.0_dp * atan(1.0_dp)

  ! arguments and paths
  if (command_argument_count() < 3 .or. command_argument_count() > 4) then
    print *, 'Usage: era5_iso YEAR MONTH DAY [OUT_DIR]'
    stop 1
  end if
  call get_command_argument(1, arg); read(arg, *) year
  call get_command_argument(2, arg); read(arg, *) month
  call get_command_argument(3, arg); read(arg, *) day
  dir_out = DIR_OUT_DEFAULT
  if (command_argument_count() == 4) call get_command_argument(4, dir_out)
  call get_env("ERA5_PL_DIR", DIR_ERA5_DEFAULT, dir_era5)
  call get_env("ERA5_ZS_FILE", FILE_ZS_DEFAULT, file_zs)
  write(yearname, '(i4.4)') year
  write(monthname, '(i2.2)') month
  write(dayname, '(i2.2)') day
  print *, yearname, monthname, dayname

  ! open the input files
  print *, "Reading: "
  do n = 1, NVAR
    file_name = trim(dir_era5) // "/" // yearname // monthname // "/e5.oper.an.pl.128_" // trim(VAR_CODE(n)) // &
         ".ll025sc." // yearname // monthname // dayname // "00_" // yearname // monthname // dayname // "23.nc"
    print *, trim(file_name)
    call check(nf90_open(file_name, NF90_NOWRITE, ncid(n)))
    call check(nf90_inq_varid(ncid(n), trim(VAR_NAME(n)), varid(n)))
  end do
  print *, trim(file_zs)
  call check(nf90_open(file_zs, NF90_NOWRITE, ncid_zs))
  call check(nf90_inq_varid(ncid_zs, "Z", varid_zs))

  ! grid
  NLON = dim_len(ncid(iT), "longitude")
  NLAT = dim_len(ncid(iT), "latitude")
  nlev_file = dim_len(ncid(iT), "level")
  ntime = dim_len(ncid(iT), "time")
  allocate(lat(NLAT), lon(NLON), pressure(nlev_file))
  call check(nf90_inq_varid(ncid(iT), "latitude", varid_c))
  call check(nf90_get_var(ncid(iT), varid_c, lat))
  call check(nf90_inq_varid(ncid(iT), "longitude", varid_c))
  call check(nf90_get_var(ncid(iT), varid_c, lon))
  call check(nf90_inq_varid(ncid(iT), "level", varid_c))
  call check(nf90_get_var(ncid(iT), varid_c, pressure))

  ! levels from P_TOP down to the bottom (ERA5 levels are in hPa, top to bottom)
  level_start = findloc(pressure >= P_TOP, .true., 1)
  NLEV = nlev_file - level_start + 1
  allocate(press(NLEV))
  press = 100.0_dp * pressure(level_start:nlev_file)

  if (mod(NLON, nlon_slice) /= 0 .or. mod(NLAT - 1, nlat_slice - 1) /= 0) then
    print *, 'Grid ', NLON, NLAT, ' does not fit the ', nlon_slice, nlat_slice, ' boxes'
    stop 1
  end if
  NX = NLON / nlon_slice
  NY = (NLAT - 1) / (nlat_slice - 1)

  ! grid-cell area, and output coordinates at the box centres
  dlat = abs(lat(2) - lat(1)) * pi / 180.0_dp
  allocate(area_row(NLAT))
  area_row = radius**2 * (2.0_dp * pi / NLON) * dlat * max(cos(pi * lat / 180.0_dp), 0.0_dp)
  weight_slc = 1.0_dp
  weight_slc(1) = 0.5_dp
  weight_slc(nlat_slice) = 0.5_dp

  allocate(lat_out(NY), lon_out(NX), the_lev(NBIN))
  do jbin = 1, NY
    lat_out(jbin) = lat((jbin - 1) * (nlat_slice - 1) + (nlat_slice + 1) / 2)
  end do
  do ibin = 1, NX
    lon_out(ibin) = sum(lon((ibin - 1) * nlon_slice + 1:ibin * nlon_slice)) / nlon_slice
  end do
  dthe = (THE_MAX - THE_MIN) / (NBIN - 1)
  do i = 1, NBIN
    the_lev(i) = THE_MIN + (i - 1) * dthe
  end do

  ! surface geopotential
  allocate(GZS(NLON, NLAT))
  call check(nf90_get_var(ncid_zs, varid_zs, GZS, [1, 1, 1], [NLON, NLAT, 1]))
  call packing(ncid_zs, varid_zs, x, weight)
  GZS = GZS * x + weight

  allocate(temp(NLON, NLAT, NLEV), sphum(NLON, NLAT, NLEV), omega(NLON, NLAT, NLEV), GZ(NLON, NLAT, NLEV))
  allocate(qliq(NLON, NLAT, NLEV), qice(NLON, NLAT, NLEV), dummy(NLON, NLAT, NLEV))
  allocate(entropy(NLON, NLAT, NLEV), enthalpy(NLON, NLAT, NLEV), theta_e(NLON, NLAT, NLEV))
  allocate(O0(NLON, NLAT, NLEV), O13(NLON, NLAT, NLEV), O410(NLON, NLAT, NLEV), OSW(NLON, NLAT, NLEV))
  allocate(above(NLON))
  allocate(ACC(NF, NBIN, NX, NY, NLEV), ACCZ(NFZ, NBIN, NY, NLEV))
  ACC = 0.0_dp
  ACCZ = 0.0_dp

  do istep = 1, ntime
    print *, "timestep :", istep

    call read_level_field(iT, temp)
    call read_level_field(iQ, sphum)
    call read_level_field(iW, omega)
    omega = omega / grav
    call read_level_field(iZ, GZ)
    sphum = max(sphum, 1e-10_dp)

    ! condensates: snow + cloud ice, rain + cloud liquid
    call read_level_field(iCSWC, qice)
    call read_level_field(iCIWC, dummy)
    qice = qice + dummy
    call read_level_field(iCRWC, qliq)
    call read_level_field(iCLWC, dummy)
    qliq = qliq + dummy

    do k = 1, NLEV
      call moist_thermo(theta_e(:,:,k), entropy(:,:,k), enthalpy(:,:,k), temp(:,:,k), sphum(:,:,k), &
           qliq(:,:,k), qice(:,:,k), press(k))
    end do

    ! zonal decomposition of omega, with zero below the surface
    do k = 1, NLEV
      do j = 1, NLAT
        above = GZ(:,j,k) > GZS(:,j)
        dummy(:,j,k) = merge(omega(:,j,k), 0.0_dp, above)
        if (count(above) > 0) then
          O0(:,j,k) = sum(dummy(:,j,k)) / count(above)
        else
          O0(:,j,k) = 0.0_dp
        end if
        O13(:,j,k) = band_pass(dummy(:,j,k), 1, 3)
        O410(:,j,k) = band_pass(dummy(:,j,k), 4, 10)
      end do
    end do
    OSW = omega - O0 - O13 - O410

    ! bin, sharing each point linearly between the two nearest theta_e levels
    do k = 1, NLEV
      do jbin = 1, NY
        do jsl = 1, nlat_slice
          j = (jbin - 1) * (nlat_slice - 1) + jsl
          do ibin = 1, NX
            do isl = 1, nlon_slice
              i = (ibin - 1) * nlon_slice + isl
              if (GZ(i,j,k) > GZS(i,j)) then
                x = (theta_e(i,j,k) - THE_MIN) / dthe
                bin_ind = min(max(ceiling(x), 1), NBIN)
                weight = min(max(bin_ind - x, 0.0_dp), 1.0_dp)
                w = area_row(j) * weight_slc(jsl) * [weight, 1.0_dp - weight]

                vals = [1.0_dp, omega(i,j,k), temp(i,j,k), sphum(i,j,k), GZ(i,j,k), entropy(i,j,k), &
                     enthalpy(i,j,k), qliq(i,j,k), qice(i,j,k), sphum(i,j,k) * omega(i,j,k), &
                     qliq(i,j,k) * omega(i,j,k), qice(i,j,k) * omega(i,j,k)]
                valz = [O0(i,j,k), O13(i,j,k), O410(i,j,k), OSW(i,j,k)]
                do l = 1, 2
                  b = min(bin_ind + l - 1, NBIN)
                  ACC(:,b,ibin,jbin,k) = ACC(:,b,ibin,jbin,k) + w(l) * vals
                  ACCZ(:,b,jbin,k) = ACCZ(:,b,jbin,k) + w(l) * valz
                end do
              end if
            end do
          end do
        end do
      end do
    end do
  end do

  do n = 1, NVAR
    call check(nf90_close(ncid(n)))
  end do
  call check(nf90_close(ncid_zs))
  deallocate(temp, sphum, omega, GZ, qliq, qice, dummy, entropy, enthalpy, theta_e, O0, O13, O410, OSW)

  ! daily means
  ACC = ACC / ntime
  ACCZ = ACCZ / ntime

  ! remove the mean vertical velocity: over each pressure level (MFC_THE), over each latitude band
  ! (wavenumber components) and over each column (MFCNV_THE)
  allocate(WNET(NLEV), WLAT(NY, NLEV), WCOL(NX, NY, NLEV), AREA_Y(NBIN, NY, NLEV))
  AREA_Y = sum(ACC(fAREA,:,:,:,:), 2)
  WNET = mean_w(sum(sum(sum(ACC(fMF,:,:,:,:), 1), 1), 1), sum(sum(sum(ACC(fAREA,:,:,:,:), 1), 1), 1))
  WCOL = mean_w(sum(ACC(fMF,:,:,:,:), 1), sum(ACC(fAREA,:,:,:,:), 1))

  allocate(MFC_THE(NBIN, NX, NY, NLEV), MFCNV_THE(NBIN, NX, NY, NLEV))
  do k = 1, NLEV
    MFC_THE(:,:,:,k) = ACC(fMF,:,:,:,k) - WNET(k) * ACC(fAREA,:,:,:,k)
    do i = 1, NBIN
      MFCNV_THE(i,:,:,k) = ACC(fMF,i,:,:,k) - WCOL(:,:,k) * ACC(fAREA,i,:,:,k)
    end do
  end do

  ! zonal mean: global mean removed; wavenumber components: latitude-band mean removed
  WNET = mean_w(sum(sum(ACCZ(1,:,:,:), 1), 1), sum(sum(AREA_Y, 1), 1))
  do k = 1, NLEV
    ACCZ(1,:,:,k) = ACCZ(1,:,:,k) - WNET(k) * AREA_Y(:,:,k)
  end do
  do n = 2, NFZ
    WLAT = mean_w(sum(ACCZ(n,:,:,:), 1), sum(AREA_Y, 1))
    do i = 1, NBIN
      ACCZ(n,i,:,:) = ACCZ(n,i,:,:) - WLAT * AREA_Y(i,:,:)
    end do
  end do

  ! streamfunctions: cumulative sums over theta_e
  allocate(STR_THE(NBIN, NLEV), STRZ_THE(NBIN, NLEV, NFZ))
  STR_THE = cumsum(sum(sum(MFC_THE, 2), 2))
  do n = 1, NFZ
    STRZ_THE(:,:,n) = cumsum(sum(ACCZ(n,:,:,:), 2))
  end do

  ! output
  file_name = trim(dir_out) // "/" // yearname // "/" // FILE_OUT // yearname // "_" // monthname // "_" // &
       dayname // ".nc"
  print *, " Outputfile: " // trim(file_name)
  call check(nf90_create(file_name, ior(NF90_CLOBBER, NF90_NETCDF4), ncid_out))

  call check(nf90_def_dim(ncid_out, "lat", NY, y_dimid))
  call check(nf90_def_dim(ncid_out, "lon", NX, x_dimid))
  call check(nf90_def_dim(ncid_out, "press", NLEV, z_dimid))
  call check(nf90_def_dim(ncid_out, "theta_e", NBIN, the_dimid))
  call check(nf90_def_dim(ncid_out, "day", NF90_UNLIMITED, rec_dimid))
  call def_coord("lat", y_dimid, "degrees_north", varid_coord(1))
  call def_coord("lon", x_dimid, "degrees_east", varid_coord(2))
  call def_coord("press", z_dimid, "Pa", varid_coord(3))
  call def_coord("theta_e", the_dimid, "K", varid_coord(4))

  do n = 1, NF
    call def_field(F_NAME(n), [the_dimid, x_dimid, y_dimid, z_dimid, rec_dimid], varid_out(n))
  end do
  call def_field("MFCNV_THE", [the_dimid, x_dimid, y_dimid, z_dimid, rec_dimid], varid_out(NF + 1))
  call check(nf90_put_att(ncid_out, varid_out(fMF), "long_name", &
       "mass flux (omega * area / g), mean over each pressure level removed"))
  call check(nf90_put_att(ncid_out, varid_out(NF + 1), "long_name", &
       "mass flux (omega * area / g), mean over each column removed"))
  do n = 1, NFZ
    call def_field(FZ_NAME(n), [the_dimid, y_dimid, z_dimid, rec_dimid], varid_z(n))
  end do
  call def_field("STR_THE", [the_dimid, z_dimid, rec_dimid], varid_str)
  do n = 1, NFZ
    call def_field(STR_NAME(n), [the_dimid, z_dimid, rec_dimid], varid_strz(n))
  end do
  call check(nf90_put_att(ncid_out, NF90_GLOBAL, "source", "ERA5 hourly pressure levels, RDA d633000"))
  call check(nf90_put_att(ncid_out, NF90_GLOBAL, "date", yearname // "-" // monthname // "-" // dayname))
  call check(nf90_enddef(ncid_out))

  call check(nf90_put_var(ncid_out, varid_coord(1), lat_out))
  call check(nf90_put_var(ncid_out, varid_coord(2), lon_out))
  call check(nf90_put_var(ncid_out, varid_coord(3), press))
  call check(nf90_put_var(ncid_out, varid_coord(4), the_lev))
  call check(nf90_put_var(ncid_out, varid_out(fAREA), ACC(fAREA,:,:,:,:), [1, 1, 1, 1, 1], [NBIN, NX, NY, NLEV, 1]))
  call check(nf90_put_var(ncid_out, varid_out(fMF), MFC_THE, [1, 1, 1, 1, 1], [NBIN, NX, NY, NLEV, 1]))
  call check(nf90_put_var(ncid_out, varid_out(NF + 1), MFCNV_THE, [1, 1, 1, 1, 1], [NBIN, NX, NY, NLEV, 1]))
  do n = fT, NF
    call check(nf90_put_var(ncid_out, varid_out(n), ACC(n,:,:,:,:), [1, 1, 1, 1, 1], [NBIN, NX, NY, NLEV, 1]))
  end do
  do n = 1, NFZ
    call check(nf90_put_var(ncid_out, varid_z(n), ACCZ(n,:,:,:), [1, 1, 1, 1], [NBIN, NY, NLEV, 1]))
    call check(nf90_put_var(ncid_out, varid_strz(n), STRZ_THE(:,:,n), [1, 1, 1], [NBIN, NLEV, 1]))
  end do
  call check(nf90_put_var(ncid_out, varid_str, STR_THE, [1, 1, 1], [NBIN, NLEV, 1]))
  call check(nf90_close(ncid_out))

contains

  subroutine check(status)
    integer, intent(in) :: status

    if (status /= nf90_noerr) then
      print *, trim(nf90_strerror(status))
      stop "Stopped"
    end if
  end subroutine check

  subroutine get_env(name, default, value)
    character(len=*), intent(in) :: name, default
    character(len=*), intent(out) :: value
    integer :: status

    call get_environment_variable(name, value, status=status)
    if (status /= 0 .or. len_trim(value) == 0) value = default
  end subroutine get_env

  integer function dim_len(nc, name)
    integer, intent(in) :: nc
    character(len=*), intent(in) :: name
    integer :: id

    call check(nf90_inq_dimid(nc, name, id))
    call check(nf90_inquire_dimension(nc, id, len=dim_len))
  end function dim_len

  ! read levels level_start:nlev_file of input n at time istep
  subroutine read_level_field(n, field)
    integer, intent(in) :: n
    real(dp), intent(out) :: field(:,:,:)

    real(dp) :: scale, offset

    call check(nf90_get_var(ncid(n), varid(n), field, [1, 1, level_start, istep], [NLON, NLAT, NLEV, 1]))
    call packing(ncid(n), varid(n), scale, offset)
    field = field * scale + offset
  end subroutine read_level_field

  ! scale_factor and add_offset of a packed variable (1 and 0 otherwise)
  subroutine packing(nc, id, scale, offset)
    integer, intent(in) :: nc, id
    real(dp), intent(out) :: scale, offset

    if (nf90_get_att(nc, id, "scale_factor", scale) /= nf90_noerr) scale = 1.0_dp
    if (nf90_get_att(nc, id, "add_offset", offset) /= nf90_noerr) offset = 0.0_dp
  end subroutine packing

  elemental real(dp) function mean_w(mf, area)
    real(dp), intent(in) :: mf, area

    mean_w = 0.0_dp
    if (area > 0.0_dp) mean_w = mf / area
  end function mean_w

  function cumsum(a) result(c)
    real(dp), intent(in) :: a(:,:)
    real(dp) :: c(size(a, 1), size(a, 2))
    integer :: i

    c(1,:) = a(1,:)
    do i = 2, size(a, 1)
      c(i,:) = c(i - 1,:) + a(i,:)
    end do
  end function cumsum

  subroutine def_coord(name, dim, units, id)
    character(len=*), intent(in) :: name, units
    integer, intent(in) :: dim
    integer, intent(out) :: id

    call check(nf90_def_var(ncid_out, name, NF90_FLOAT, [dim], id))
    call check(nf90_put_att(ncid_out, id, "units", units))
  end subroutine def_coord

  subroutine def_field(name, dims, id)
    character(len=*), intent(in) :: name
    integer, intent(in) :: dims(:)
    integer, intent(out) :: id

    call check(nf90_def_var(ncid_out, trim(name), NF90_FLOAT, dims, id, deflate_level=1))
  end subroutine def_field

end program era5_iso
