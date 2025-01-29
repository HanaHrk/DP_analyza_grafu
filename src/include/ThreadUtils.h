#pragma once

#if WIN32

#include <windows.h>
#include <tlhelp32.h>
#include <iostream>

DWORD get_number_of_threads(DWORD process_id) {
    HANDLE h_thread_snap = INVALID_HANDLE_VALUE;
    THREADENTRY32 te32;
    DWORD thread_count = 0;
    h_thread_snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (h_thread_snap == INVALID_HANDLE_VALUE) {
        return 0;
    }
    te32.dwSize = sizeof(THREADENTRY32);
    if (!Thread32First(h_thread_snap, &te32)) {
        CloseHandle(h_thread_snap); // Must clean up the snapshot object!
        return 0;
    }
    do {
        if (te32.th32OwnerProcessID == process_id) {
            thread_count++;
        }
    } while (Thread32Next(h_thread_snap, &te32));
    CloseHandle(h_thread_snap);
    return thread_count;
}

DWORD get_number_of_threads_current() {
    DWORD process_id = GetCurrentProcessId();
    return get_number_of_threads(process_id);
}

#endif
