// Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
#ifndef __DACE_ALLOC_H
#define __DACE_ALLOC_H

#include <cstddef>
#include <algorithm>
#include <new>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "types.h"

namespace dace
{
    class MemoryPool
    {
        public:
            MemoryPool() = default;
            MemoryPool(const MemoryPool &) = delete;
            MemoryPool &operator=(const MemoryPool &) = delete;

            ~MemoryPool()
            {
                for (const auto &block : available)
                    free_block(block);
                for (const auto &block : active)
                    free_block(block.second);
            }

            template <typename T>
            T *allocate(std::size_t count, std::size_t alignment)
            {
                if (count > static_cast<std::size_t>(-1) / sizeof(T))
                    throw std::bad_array_new_length();

                const std::size_t size = std::max<std::size_t>(count * sizeof(T), 1);
                alignment = std::max({alignment, alignof(T), alignof(std::max_align_t)});
                Block block = acquire(size, alignment);
                T *result = static_cast<T *>(block.pointer);
                ConstructionGuard<T> guard(block, result);
                for (std::size_t constructed = 0; constructed < count; ++constructed)
                    guard.construct_one();

                std::lock_guard<std::mutex> lock(mutex);
                if (!active.emplace(result, block).second)
                    throw std::logic_error("Memory pool returned an active block");
                guard.dismiss();
                return result;
            }

            template <typename T>
            void release(T *pointer, std::size_t count)
            {
                if (pointer == nullptr)
                    return;

                Block block;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    auto active_block = active.find(pointer);
                    if (active_block == active.end())
                        throw std::invalid_argument("Pointer was not allocated by this memory pool");
                    block = active_block->second;
                    active.erase(active_block);
                }

                for (std::size_t i = count; i > 0; --i)
                    pointer[i - 1].~T();

                BlockGuard guard(block);
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    available.push_back(block);
                }
                guard.dismiss();
            }

        private:
            struct Block
            {
                void *pointer;
                std::size_t size;
                std::size_t alignment;
            };

            class BlockGuard
            {
              public:
                explicit BlockGuard(const Block &block) : block_(block) {}
                BlockGuard(const BlockGuard &) = delete;
                BlockGuard &operator=(const BlockGuard &) = delete;

                ~BlockGuard()
                {
                    if (armed_)
                        free_block(block_);
                }

                void dismiss() { armed_ = false; }

              private:
                Block block_;
                bool armed_ = true;
            };

            template <typename T>
            class ConstructionGuard
            {
              public:
                ConstructionGuard(const Block &block, T *elements) : block_(block), elements_(elements) {}
                ConstructionGuard(const ConstructionGuard &) = delete;
                ConstructionGuard &operator=(const ConstructionGuard &) = delete;

                ~ConstructionGuard()
                {
                    if (!armed_)
                        return;
                    while (constructed_ > 0)
                        elements_[--constructed_].~T();
                    free_block(block_);
                }

                void construct_one()
                {
                    ::new (static_cast<void *>(elements_ + constructed_)) T;
                    ++constructed_;
                }

                void dismiss() { armed_ = false; }

              private:
                Block block_;
                T *elements_;
                std::size_t constructed_ = 0;
                bool armed_ = true;
            };

            Block acquire(std::size_t size, std::size_t alignment)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    auto best = available.end();
                    for (auto block = available.begin(); block != available.end(); ++block) {
                        if (block->size >= size && block->alignment >= alignment
                            && (best == available.end() || block->size < best->size))
                            best = block;
                    }
                    if (best != available.end()) {
                        Block result = *best;
                        available.erase(best);
                        return result;
                    }
                }

                return {allocate_block(size, alignment), size, alignment};
            }

            static void *allocate_block(std::size_t size, std::size_t alignment)
            {
    #if defined(__cpp_aligned_new)
                if (alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
                    return ::operator new(size, std::align_val_t(alignment));
    #else
                (void)alignment;
    #endif
                return ::operator new(size);
            }

            static void free_block(const Block &block)
            {
    #if defined(__cpp_aligned_new)
                if (block.alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__) {
                    ::operator delete(block.pointer, std::align_val_t(block.alignment));
                    return;
                }
    #endif
                ::operator delete(block.pointer);
            }

            std::mutex mutex;
            std::vector<Block> available;
            std::unordered_map<void *, Block> active;
    };

    // Aligned heap arrays. The aligned ``operator delete[]`` runs no destructors, so only trivially
    // destructible types are allocated aligned; all others use plain ``new[]`` / ``delete[]``.

    template <typename T>
    DACE_HDFI T *aligned_new_array(std::size_t size, std::size_t alignment)
    {
#if defined(__cpp_aligned_new)
        // Compiler supports aligned new (C++17 feature)
        if constexpr (std::is_trivially_destructible<T>::value) {
            return new (std::align_val_t(alignment)) T[size];
        } else {
            return new T[size];
        }
#else
        // Plain new and delete[], just to be safe
        return new T[size];
#endif
    }

    template <typename T>
    DACE_HDFI void aligned_delete_array(T *ptr, std::size_t alignment)
    {
#if defined(__cpp_aligned_new)
        // Compiler supports aligned new (C++17 feature)
        if constexpr (std::is_trivially_destructible<T>::value) {
            ::operator delete[](ptr, std::align_val_t(alignment));
        } else {
            delete[] ptr;
        }
#else
        // Plain new and delete[], just to be safe
        delete[] ptr;
#endif
    }
}  // namespace dace

#endif  // __DACE_ALLOC_H
