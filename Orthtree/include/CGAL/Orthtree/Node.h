// Copyright (c) 2007-2020  INRIA (France).
// All rights reserved.
//
// This file is part of CGAL (www.cgal.org).
//
// $URL$
// $Id$
// SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial
//
// Author(s)     : Jackson Campolattaro, Cédric Portaneri, Tong Zhao

#ifndef CGAL_ORTHTREE_NODE_H
#define CGAL_ORTHTREE_NODE_H

#include <CGAL/license/Orthtree.h>

#include <CGAL/Orthtree.h>

#include <boost/range/iterator_range.hpp>
#include <boost/core/span.hpp>

#include <array>
#include <memory>
#include <bitset>
#include <cassert>
#include <iostream>

namespace CGAL {

/*!

  \brief represents a single node of the tree. Alternatively referred
  to as a cell, orthant, or sub-tree.

  A `Node` is a lightweight object and thus generally passed by
  copy. It is also a model of `ConstRange` with value type `Traits::Point_d`.

  \cgalModels `ConstRange`
 */
template <typename Traits, typename PointRange, typename PointMap>
class Orthtree<Traits, PointRange, PointMap>::Node {

public:

  /// \name Types
  /// @{

  typedef Orthtree<Traits, PointRange, PointMap> Enclosing; ///< Orthtree type (enclosing class).
  typedef typename Enclosing::Dimension Dimension; ///< Dimension type.
  typedef typename Enclosing::Degree Degree; ///< Degree type.
  typedef typename Enclosing::Node_index Node_index; ///< Index type.
  typedef typename Enclosing::Maybe_node_index Maybe_node_index; ///< Index type.
  typedef std::vector<std::size_t> Polygon_index_range;

  /*!
    \brief Self typedef for convenience.
   */
  typedef typename Orthtree<Traits, PointRange, PointMap>::Node Self;

  /*!
    \brief Set of bits representing this node's relationship to its parent.

    Equivalent to an array of Booleans, where index[0] is whether `x`
    is greater, index[1] is whether `y` is greater, index[2] is whether
    `z` is greater, and so on for higher dimensions if needed.
    Used to represent a node's relationship to the center of its parent.
   */
  typedef std::bitset<Dimension::value> Local_coordinates;

  /*!
    \brief Coordinates representing this node's relationship
    with the rest of the tree.

    Each value `(x, y, z, ...)` of global coordinates is calculated by doubling
    the parent's global coordinates and adding the local coordinates.
   */
  typedef std::array<std::uint32_t, Dimension::value> Global_coordinates;


  typedef typename PointRange::const_iterator const_iterator; ///< constant iterator type.

  /*!
    \brief Adjacency directions.
   */
  typedef typename Traits::Adjacency Adjacency;

  /// @}

private:

  /// \cond SKIP_IN_MANUAL
  /*!
    \brief point range type.
   */
  typedef typename PointRange::iterator iterator; ///< constant iterator type.
  typedef boost::iterator_range<iterator> Point_range;
  /// \endcond

  Point_range m_points;
  Polygon_index_range m_polygons;
  std::vector<std::vector<std::pair<std::size_t, Point> > > m_clipped_polygons;
  std::uint8_t m_depth = 0;
  Global_coordinates m_global_coordinates{};

  Maybe_node_index m_parent_index{};
  Maybe_node_index m_children_index{};

  // Only the Orthtree class has access to the non-default
  // constructor, mutators, etc.
  friend Enclosing;

public:

  /// \name Construction
  /// @{

  /// \cond SKIP_IN_MANUAL
  Node() = default; // constructs a root node
  /// \endcond

  /*!
    \brief creates a new node, optionally as the child of a parent

    If no parent is provided, the node created is assumed to be the
    root of a tree.  This means that `parent.is_null()` returns
    `true`, and the depth is zero.  If a parent is provided, the node
    becomes the child of that parent.  In that case, an index should
    be passed, telling this node its relationship to its parent.
    Depth and global coordinates are automatically determined in the
    constructor, and should generally be considered immutable after
    construction.

    \param parent the node containing this one
    \param index this node's relationship to its parent
  */
  explicit Node(Node_index parent_index, Global_coordinates parent_coordinates,
                std::size_t depth, Local_coordinates local_coordinates) :
    m_parent_index(parent_index), m_depth(depth) {

    for (int i = 0; i < Dimension::value; i++)
      m_global_coordinates[i] = (2 * parent_coordinates[i]) + local_coordinates[i];

  }

  /// @}

public:

  /// \name Member Access
  /// @{

  /*!
   * \brief Access to the content held by this node
   * \return a reference to the collection of point indices
   */
  Point_range& points() { return m_points; }

  const Point_range& points() const { return m_points; }

  Polygon_index_range& polygons() { return m_polygons; }

  const Polygon_index_range& polygons() const { return m_polygons; }

  std::vector<std::vector<std::pair<std::size_t, Point> > >& clipped_polygons() { return m_clipped_polygons; }

  const std::vector<std::vector<std::pair<std::size_t, Point> > >&clipped_polygons() const { return m_clipped_polygons; }

  /// @}

  /// \name Type & Location
  /// @{

  /*!
    \brief returns `true` if the node has no parent, `false` otherwise.
    \pre `!is_null()`
  */
  bool is_root() const {
    return !m_parent_index.has_value();
  }

  /*!
    \brief returns `true` if the node has no children, `false` otherwise.
    \pre `!is_null()`
  */
  bool is_leaf() const {
    return !m_children_index.has_value();
  }

  /*!
    \brief returns this node's depth.
    \pre `!is_null()`
  */
  std::uint8_t depth() const {
    return m_depth;
  }

  /*!
    \brief returns this node's local coordinates (in relation to its parent).
    \pre `!is_null()`
  */
  Local_coordinates local_coordinates() const {

    Local_coordinates result;

    for (std::size_t i = 0; i < Dimension::value; ++i)
      result[i] = global_coordinates()[i] & 1;

    return result;
  }

  /*!
    \brief returns this node's global coordinates.
    \pre `!is_null()`
  */
  Global_coordinates global_coordinates() const {
    return m_global_coordinates;
  }

  /// @}

  /// \name Point Range
  /// @{

  /*!
    \brief returns the number of points of this node.
   */
  std::size_t size() const {
    if (m_polygons.size())
      return std::size_t(std::distance(m_polygons.begin(), m_polygons.end()));
    else
      return std::size_t(std::distance(m_points.begin(), m_points.end()));
  }

  /// @}

  /// \cond SKIP_IN_MANUAL
  /// \name Operators
  /// @{

  /*!
   * \brief compares the topology of this node to another node.
   *
   * \param rhs node to compare with
   * \return whether the nodes have different topology.
   */
  bool operator==(const Self& rhs) const {
    return true;
  }

  friend std::ostream& operator<<(std::ostream& os, const Self& node) {
    return internal::print_orthtree_node(os, node);
  }
  /// \endcond
};

}

#endif //CGAL_ORTHTREE_NODE_H
