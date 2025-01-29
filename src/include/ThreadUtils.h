#pragma once

#if WIN32

#include <windows.h>
#include <tlhelp32.h>
#include <iostream>

inline DWORD get_number_of_threads(const DWORD process_id)
{
    auto h_thread_snap = INVALID_HANDLE_VALUE;
    THREADENTRY32 te32;
    DWORD thread_count = 0;
    h_thread_snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (h_thread_snap == INVALID_HANDLE_VALUE)
    {
        return 0;
    }
    te32.dwSize = sizeof(THREADENTRY32);
    if (!Thread32First(h_thread_snap, &te32))
    {
        CloseHandle(h_thread_snap); // Must clean up the snapshot object!
        return 0;
    }
    do
    {
        if (te32.th32OwnerProcessID == process_id)
        {
            thread_count++;
        }
    }
    while (Thread32Next(h_thread_snap, &te32));
    CloseHandle(h_thread_snap);
    return thread_count;
}

inline DWORD get_number_of_threads_current()
{
    const DWORD process_id = GetCurrentProcessId();
    return get_number_of_threads(process_id);
}

inline std::thread find_thread_count(std::atomic_bool& stop)
{
    return std::thread([&stop]
    {
        int max_thread_count = 0;
        while (stop.load() == false)
        {
            const auto thread_count = get_number_of_threads_current();
            if (thread_count > max_thread_count)
            {
                max_thread_count = static_cast<int>(thread_count);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cout << "Max thread count: " << max_thread_count << std::endl;
    });
}

#endif
