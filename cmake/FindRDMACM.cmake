# CMakeLists.txt.
#
# @author Steffen Vogel <stvogel@eonerc.rwth-aachen.de>
# @copyright 2018, Institute for Automation of Complex Power Systems, EONERC
# @license GNU General Public License (version 3)
#
# VILLASnode
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
###################################################################################

find_path(RDMACM_INCLUDE_DIR
	NAMES rdma/rdma_cma.h
)

find_library(RDMACM_LIBRARY
	NAMES rdmacm
)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set VILLASNODE_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(RDMACM DEFAULT_MSG
	RDMACM_LIBRARY RDMACM_INCLUDE_DIR)

mark_as_advanced(RDMACM_INCLUDE_DIR RDMACM_LIBRARY)

set(RDMACM_LIBRARIES ${RDMACM_LIBRARY})
set(RDMACM_INCLUDE_DIRS ${RDMACM_INCLUDE_DIR})

add_library(RDMACM INTERFACE)
target_include_directories(RDMACM SYSTEM INTERFACE ${RDMACM_INCLUDE_DIRS})
target_link_libraries(RDMACM INTERFACE ${RDMACM_LIBRARIES})
