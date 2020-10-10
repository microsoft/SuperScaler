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

find_path(IBVERBS_INCLUDE_DIR
	NAMES infiniband/verbs.h
)

find_library(IBVERBS_LIBRARY
	NAMES ibverbs
)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set VILLASNODE_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(IBVerbs DEFAULT_MSG
	IBVERBS_LIBRARY)

mark_as_advanced(IBVERBS_INCLUDE_DIR IBVERBS_LIBRARY)

set(IBVERBS_LIBRARIES ${IBVERBS_LIBRARY})
set(IBVERBS_INCLUDE_DIRS ${IBVERBS_INCLUDE_DIR})

add_library(IBVerbs INTERFACE)
target_include_directories(IBVerbs SYSTEM INTERFACE ${IBVERBS_INCLUDE_DIRS})
target_link_libraries(IBVerbs INTERFACE ${IBVERBS_LIBRARIES})

